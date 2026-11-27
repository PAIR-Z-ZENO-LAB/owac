"""Recorded stereo inputs and time-indexed native command chunks for a single robot."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from owac.raw.all_bench_distillation import load_inventory, read_npz_member
from owac.raw.teleop_mcap import CAMERAS
from owac.training.compact_actions import relative_actions
from owac.training.fruit_grasp import grasp_phase
from owac.training.fruit_placement import placement_events, placement_rows
from owac.training.raw_measurements import attenuate_measurement
from owac.training.raw_ssl import RawEncoder
from owac.training.teleop_probe import WALL


def command_chunks(observed, command_ns, commands, horizon, frequency):
    offsets = np.rint(np.arange(horizon) * 1e9 / frequency).astype(np.int64)
    index = np.searchsorted(command_ns, observed[:, None] + offsets)
    return commands[np.minimum(index, len(commands) - 1)], index < len(commands)


def episode_labels(src, cfg):
    with np.load(src.path, allow_pickle=False) as z:
        observed, state = z["observation_ns"], z["state"]
        if (z["state_ns"] > observed).any() or (z["state_log_ns"] > observed).any():
            raise ValueError(f"Noncausal follower state in {src.group_id}")
        actions, valid = command_chunks(
            observed,
            z["command_ns"],
            z["commands"],
            cfg["model"]["action_horizon"],
            cfg["observation"]["frequency_hz"],
        )
    relative = relative_actions(actions, state, cfg["observation"]["relative_action_mask"])
    return state, actions, relative, valid


def fit_statistics(cfg, sources):
    states, labels = [], []
    for src in sources:
        state, _, relative, valid = episode_labels(src, cfg)
        states.append(state)
        labels.append(relative[valid])
    states = np.concatenate(states)
    low, high = np.quantile(np.concatenate(labels), [0.01, 0.99], axis=0)
    return dict(
        state_mean=states.mean(0).tolist(),
        state_std=states.std(0).tolist(),
        action_q01=low.tolist(),
        action_q99=high.tolist(),
    )


def cache_paths(cfg, src):
    file = Path(cfg["data"]["feature_cache"]) / (src.group_id + ".npy")
    return file, file.with_suffix(".json")


def raw_views(src):
    return [read_npz_member(src.path, k + ".npy") for k in CAMERAS]


def frozen_teleop_encoder(cfg, device):
    checkpoint = Path(cfg["assets"]["visual_encoder"])
    with checkpoint.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != cfg["assets"]["visual_sha256"]:
            raise ValueError("Frozen visual checkpoint differs from the configured encoder")
    encoder = RawEncoder(cfg["model"]["raw_encoder"]).to(device).eval().requires_grad_(False)
    saved = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    encoder.load_state_dict(saved["encoder"], strict=True)
    return encoder, saved["step"]


@torch.no_grad()
def encode_stereo_block(encoder, views, start, stop, device):
    raw = np.stack([view[start:stop] for view in views], axis=1)
    frames = torch.from_numpy(raw.view(np.int16)).to(device=device, dtype=torch.float32)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        _, tokens = encoder(frames.flatten(0, 1))
    return tokens.unflatten(0, (stop - start, len(views))).to(torch.bfloat16)


def gripper_travel(actions, valid):
    """Recorded aperture change ending at a command, restricted to real adjacent labels."""
    widths = actions[..., 7]
    motion = (widths[:, 1:] - widths[:, :-1]).abs() * (valid[:, 1:] & valid[:, :-1])
    return torch.cat((torch.zeros_like(widths[:, :1]), motion), dim=1)


def fruit_exposure(raw, cfg, stops, generator):
    sensor = cfg["model"]["raw_encoder"]["measurement_encoder"]["sensor"]
    return attenuate_measurement(raw, stops, sensor, generator)


def paired_low_light(raw, sensor, fit, generator):
    """Make a dim counterpart of each source image pair in native Bayer coordinates."""
    x0, y0, x1, y1 = WALL
    wall = raw[:, 0, y0:y1, x0:x1]
    green = torch.cat([wall[:, ::2, 1::2].flatten(1), wall[:, 1::2, ::2].flatten(1)], dim=1)
    black = sensor["black_level"]
    ix = torch.multinomial(fit["counts"], len(raw), replacement=True, generator=generator)
    gain = ((fit["levels"][ix] - black) / (green.median(1).values - black).clamp_min(1)).clamp(0, 1)
    gain = gain.reshape(-1, 1, 1, 1)
    signal = (raw - black) * gain + black
    noise = torch.randn(raw.shape, device=raw.device, generator=generator)
    for y in range(2):
        for x in range(2):
            noise[:, :, y::2, x::2] *= fit["sigma"][None, :, 2 * y + x, None, None]
    return (signal + noise * (1 - gain.square()).sqrt()).round().clamp(0, sensor["white_level"])


class FruitFrames:
    """Episode-local labels and a device-resident view bank for the 20 recorded trajectories."""

    def __init__(self, sources, cfg, device):
        self.sources, self.cfg, self.device = sources, cfg, device
        signals = [episode_labels(s, cfg) for s in sources]
        state, absolute, relative, valid = (np.concatenate(a) for a in zip(*signals, strict=True))
        self.sizes = np.array([len(row[0]) for row in signals])
        self.starts = np.r_[0, self.sizes.cumsum()[:-1]]
        self.episode = np.repeat(np.arange(len(sources)), self.sizes)
        rows = np.arange(len(state)) - self.starts[self.episode]
        self.wrong = self.tensor(
            self.starts[self.episode]
            + (rows + self.sizes[self.episode] // 2) % self.sizes[self.episode]
        )
        stats = cfg["statistics"]
        self.state = self.tensor(state)
        self.absolute = self.tensor(absolute)
        self.valid = self.tensor(valid)
        self.low = self.tensor(np.asarray(stats["action_q01"], np.float32))
        self.scale = self.tensor(np.asarray(stats["action_q99"], np.float32)) - self.low + 1e-6
        self.normalized = self.tensor(relative) * (2 / self.scale) - (1 + 2 * self.low / self.scale)
        self.state_norm = (
            self.state - self.tensor(np.asarray(stats["state_mean"], np.float32))
        ) / (self.tensor(np.asarray(stats["state_std"], np.float32)) + 1e-6)
        self.relative_mask = self.tensor(np.asarray(cfg["observation"]["relative_action_mask"]))
        self.tokens = self.tensor(np.array(cfg["task_token_ids"], np.int64))
        self.token_mask = self.tensor(np.array(cfg["task_token_mask"], bool))
        self.image_bank = None
        self.exposures = {}
        self.banana_mass = self.banana_visible = None
        if cfg["model"].get("banana_grounding") and cfg["training"].get("banana_supervision", True):
            masses, visibility = [], []
            for src in sources:
                with np.load(
                    Path(cfg["data"]["banana_labels"]) / (src.group_id + ".npz")
                ) as labels:
                    masses.append(labels["mass"].reshape(len(labels["weight"]), -1))
                    visibility.append(labels["weight"])
            self.banana_mass = self.tensor(np.concatenate(masses))
            self.banana_visible = self.tensor(np.concatenate(visibility))
        self.mode = None
        self.dark_fit = None
        if spec := cfg["training"].get("paired_low_light"):
            fit = json.loads(Path(spec["fit"]).read_text())
            self.dark_fit = {
                k: self.tensor(np.asarray(fit[name], dtype=np.float32))
                for k, name in (
                    ("levels", "wall_levels_dn"),
                    ("counts", "wall_counts"),
                    ("sigma", "noise_sigma_dn"),
                )
            }
            suites = np.asarray([src.suite for src in sources])
            self.source_light = self.tensor(np.isin(suites[self.episode], spec["source_subsets"]))
        self.motion_cdf = []
        self.approach_rows = []
        self.placement_rows = []
        if cfg["training"].get("balance_pick_and_place") or cfg["training"].get("placement_only"):
            self.placement_rows = [
                placement_rows(placement_events(s), cfg["training"]["approach_seconds"])
                for s in sources
            ]
        if cfg["training"].get("approach_fraction", 0):
            masks = [grasp_phase(s, cfg["training"]["approach_seconds"])[1] for s in sources]
            self.approach_rows = [np.flatnonzero(m) for m in masks]
        if cfg["training"].get("grip_motion_fraction", 0):
            movement = gripper_travel(self.absolute, self.valid).sum(1).cpu().numpy()
            for start, size in zip(self.starts, self.sizes, strict=True):
                cdf = movement[start : start + size].cumsum(dtype=np.float64)
                self.motion_cdf.append(cdf / cdf[-1] if cdf[-1] else None)

    def tensor(self, values):
        return torch.as_tensor(values, device=self.device)

    def load_views(self, mode):
        if self.mode == mode:
            return
        self.image_bank = None
        self.exposures.clear()
        cfg = self.cfg
        height, width = cfg["observation"]["raw_shape"]
        patch = cfg["model"]["raw_encoder"]["patch_size"]
        feature = mode == "features"
        grid = (height // patch) * (width // patch)
        shape = (2, grid, cfg["model"]["raw_encoder"]["width"]) if feature else (2, height, width)
        bank = torch.empty(
            (len(self.episode), *shape),
            device=self.device,
            dtype=torch.bfloat16 if feature else torch.int16,
        )
        encoder = None
        if feature and cfg["data"].get("features_in_memory"):
            encoder, _ = frozen_teleop_encoder(cfg, self.device)
        for start, size, src in zip(self.starts, self.sizes, self.sources, strict=True):
            if feature and encoder is None:
                path, meta = cache_paths(cfg, src)
                sig = json.loads(meta.read_text())
                if (
                    sig["encoder"] != cfg["assets"]["visual_sha256"]
                    or sig["dataset_revision"] != cfg["data"]["dataset_revision"]
                    or sig["episode"] != src.group_id
                ):
                    raise ValueError(f"Stale visual features for {src.group_id}")
                data = np.load(path, mmap_mode="r")
            else:
                views = raw_views(src)
            stride = 4 if encoder is not None else 32
            for i in range(0, size, stride):
                end = min(i + stride, size)
                if encoder is not None:
                    block = encode_stereo_block(encoder, views, i, end, self.device)
                elif feature:
                    block = torch.from_numpy(data[i:end].view(np.int16).copy()).view(torch.bfloat16)
                else:
                    block = torch.from_numpy(
                        np.stack([v[i:end] for v in views], axis=1).view(np.int16)
                    )
                bank[start + i : start + end].copy_(block)
        self.image_bank, self.mode = bank, mode

    def exposure_view(self, stops):
        if stops not in self.exposures:
            if self.mode != "raw":
                raise ValueError("Exposure evaluation requires native sensor data")
            generator = torch.Generator(device=self.device).manual_seed(
                self.cfg["evaluation"].get("exposure_seed", 20261120)
            )
            images = torch.empty_like(self.image_bank)
            for start in range(0, len(images), 16):
                at = slice(start, start + 16)
                images[at] = fruit_exposure(self.image_bank[at], self.cfg, stops, generator)
            self.exposures[stops] = images
        return self.exposures[stops]

    def batch(self, ix, *, wrong=False, exposure=0):
        n = len(ix)
        batch = dict(
            state=self.state_norm[ix],
            actions=self.normalized[ix],
            action_valid=self.valid[ix],
            tokens=self.tokens.expand(n, -1),
            token_mask=self.token_mask.expand(n, -1),
            task_id=torch.zeros(n, dtype=torch.long, device=self.device),
        )
        if self.image_bank is not None:
            images = self.exposure_view(exposure) if exposure else self.image_bank
            v = images[self.wrong[ix] if wrong else ix]
            batch[self.mode] = v.float() if self.mode == "raw" else v
        if self.banana_mass is not None:
            batch["banana_mass"] = self.banana_mass[ix].float()
            batch["banana_visible"] = self.banana_visible[ix]
        return batch

    def sample_indices(self, seed, batch):
        rng = np.random.default_rng(seed)
        ep = rng.integers(len(self.sources), size=batch)
        rows = rng.integers(self.sizes[ep])
        if self.placement_rows:
            # Slots are whole episode, pickup, transport, and opening at the basket.
            first = 2 if self.cfg["training"].get("placement_only") else 0
            selected = rng.integers(first, 4, size=batch)
            for i, (episode, phase) in enumerate(zip(ep, selected, strict=True)):
                if phase:
                    pool = self.placement_rows[episode][phase - 1]
                    if len(pool):
                        rows[i] = rng.choice(pool)
            return self.tensor(self.starts[ep] + rows)
        if self.motion_cdf:
            biased = rng.random(batch) < self.cfg["training"]["grip_motion_fraction"]
            for episode in np.unique(ep[biased]):
                mask = biased & (ep == episode)
                cdf = self.motion_cdf[episode]
                if cdf is not None:
                    rows[mask] = np.searchsorted(cdf, rng.random(mask.sum()), side="right")
        if self.approach_rows:
            near = rng.random(batch) < self.cfg["training"]["approach_fraction"]
            for episode in np.unique(ep[near]):
                mask = near & (ep == episode)
                rows[mask] = rng.choice(self.approach_rows[episode], int(mask.sum()))
        return self.tensor(self.starts[ep] + rows)

    def training_batch(self, ix, seed):
        batch = self.batch(ix)
        aug = self.cfg["training"].get("exposure_augmentation")
        if aug:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            shape = (len(ix), 1, 1, 1)
            apply = torch.rand(shape, generator=generator, device=self.device) < aug["probability"]
            stops = torch.rand(shape, generator=generator, device=self.device) * aug["max_stops"]
            batch["raw"] = fruit_exposure(batch["raw"], self.cfg, stops * apply, generator)
        return batch

    def append_light_pairs(self, batch, ix, seed):
        rows = self.source_light[ix].nonzero().flatten()
        if len(rows):
            generator = torch.Generator(device=self.device).manual_seed(seed)
            sensor = self.cfg["model"]["raw_encoder"]["measurement_encoder"]["sensor"]
            for key, value in batch.items():
                extra = value[rows]
                if key == "raw":
                    extra = paired_low_light(extra, sensor, self.dark_fit, generator)
                batch[key] = torch.cat((value, extra))
        return torch.cat((ix, ix[rows])), rows

    def physical(self, prediction, indices):
        return (
            (prediction + 1) * self.scale / 2
            + self.low
            + (self.state[indices] * self.relative_mask)[:, None, :]
        )

    def linear_baseline(self):
        x = self.state_norm.cpu().numpy()
        x = np.c_[x, np.ones(len(x), dtype=np.float32)]
        y, valid = self.normalized.cpu().numpy(), self.valid.cpu().numpy()
        weights = [
            np.linalg.lstsq(x[valid[:, k]], y[valid[:, k], k], rcond=None)[0]
            for k in range(y.shape[1])
        ]
        return self.tensor(np.stack(weights))


@torch.inference_mode()
def cache_features(cfg):
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    net, source_step = frozen_teleop_encoder(cfg, device)
    for src in load_inventory(cfg["data"]["inventory"])[0][rank::world]:
        path, meta = cache_paths(cfg, src)
        signature = dict(
            encoder=cfg["assets"]["visual_sha256"],
            dataset_revision=cfg["data"]["dataset_revision"],
            episode=src.group_id,
            step=source_step,
            cameras=list(CAMERAS),
        )
        if path.exists() and meta.exists() and json.loads(meta.read_text()) == signature:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        a, b = raw_views(src)
        n, height, width = a.shape
        patch = cfg["model"]["raw_encoder"]["patch_size"]
        shape = (n, 2, (height // patch) * (width // patch), cfg["model"]["raw_encoder"]["width"])
        partial = path.with_suffix(".partial.npy")
        array = np.lib.format.open_memmap(partial, "w+", dtype=np.uint16, shape=shape)
        for start in range(0, n, 8):
            frames = np.stack([v[start : start + 8] for v in (a, b)], axis=1)
            raw = torch.from_numpy(frames.view(np.int16)).to(device).float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, tokens = net(raw.flatten(0, 1))
            tokens = tokens.reshape(len(raw), *shape[1:]).to(torch.bfloat16).contiguous()
            array[start : start + len(raw)] = tokens.view(torch.uint16).cpu().numpy()
        array.flush()
        del array
        partial.replace(path)
        meta.write_text(json.dumps(signature))
        print(f"{src.group_id} {shape}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    cache_features(json.loads(parser.parse_args().config.read_text()))
