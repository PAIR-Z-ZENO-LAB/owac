"""Training-only RAW feature retention with an immutable, self-owned policy encoder."""

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from owac.training.compact_augmentation import shift_raw_views
from owac.training.j5_action_data import geometric_augment
from owac.training.raw_exposure import lower_exposure
from owac.training.raw_measurements import attenuate_measurement
from owac.training.raw_ssl import RawEncoder


def training_views(raw, rng, cfg, source, index, *, return_geometry=False):
    """Share crop/rotation between exposures; add sensor noise before resampling.

    The separate pair RNG leaves the action-data sampling and original view
    augmentation identical to ordinary continuation. Non-Normal samples retain
    their original exposure instead of further degrading already weak evidence.
    Optional integer shifts follow the existing geometry and are shared by the
    exposure pair. The caller reuses the local RNG across a history's frames.
    """
    retention = cfg.get("visual_retention") or cfg.get("action_exposure")
    sensor = (
        cfg.get("model", {}).get("raw_encoder", {}).get("measurement_encoder", {}).get("sensor")
    )
    degraded = None
    geometry = (
        torch.eye(3, device=raw.device, dtype=raw.dtype).expand(len(raw), 3, 3).clone()
        if return_geometry
        else None
    )
    if retention:
        pair_rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], index, 1]))
        degraded = raw.clone()
        if source.lighting == "Normal":
            if retention.get("spatial_exposure", False):
                # A smooth per-camera optical attenuation field preserves scene
                # geometry while changing both brightness and shot/read evidence.
                corners = pair_rng.uniform(0, retention["max_stops"], (len(raw), 1, 2, 2))
                stops = F.interpolate(
                    torch.tensor(corners, dtype=raw.dtype, device=raw.device),
                    size=raw.shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
            else:
                stops = float(pair_rng.uniform(0, retention["max_stops"]))
            generator = torch.Generator().manual_seed(int(pair_rng.integers(2**63)))
            degraded = (
                attenuate_measurement(raw, stops, sensor, generator)
                if sensor
                else lower_exposure(raw, stops, retention["noise"], generator)
            )
    native_grid = sensor and (sensor["layout"] in ("mono", "bayer") or sensor["units"] == "dn")
    if source.domain == "libero" and not native_grid:
        paired_geometry = copy.deepcopy(rng)
        if return_geometry:
            raw[0], geometry[0] = geometric_augment(raw[0], rng, return_transform=True)
        else:
            raw[0] = geometric_augment(raw[0], rng)
        if degraded is not None:
            degraded[0] = geometric_augment(degraded[0], paired_geometry)
    padding = cfg.get("training", {}).get("raw_shift_pixels", 0)
    if padding:
        bayer = sensor and sensor["layout"] == "bayer"
        if bayer and padding % 2:
            raise ValueError("Bayer shifts use complete 2x2 sampling cells")
        offsets = (
            2 * rng.integers(0, padding + 1, size=(len(raw), 2))
            if bayer
            else rng.integers(0, 2 * padding + 1, size=(len(raw), 2))
        )
        mode = "reflect" if bayer else "replicate"
        if return_geometry:
            translation = (
                torch.eye(3, device=raw.device, dtype=raw.dtype).expand(len(raw), 3, 3).clone()
            )
            translation[:, 0, 2] = (
                torch.as_tensor(padding - offsets[:, 1], device=raw.device) / raw.shape[-1]
            )
            translation[:, 1, 2] = (
                torch.as_tensor(padding - offsets[:, 0], device=raw.device) / raw.shape[-2]
            )
            geometry = translation @ geometry
        raw = shift_raw_views(raw, offsets, padding, padding_mode=mode)
        if degraded is not None:
            degraded = shift_raw_views(degraded, offsets, padding, padding_mode=mode)
    return (raw, degraded, geometry) if return_geometry else (raw, degraded)


def feature_distance(student, target):
    return (1 - F.cosine_similarity(student.float(), target.detach().float(), dim=-1)).mean()


def make_teacher(cfg, device):
    """The teacher stays outside policy/optimizer state and never serves actions."""
    with torch.random.fork_rng(devices=[]):
        teacher = RawEncoder(cfg["model"]["raw_encoder"])
    payload = torch.load(
        cfg["visual_retention"]["teacher_checkpoint"],
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    weights = payload["ema"] if "ema" in payload else payload["model"]
    prefix = "raw_encoder."
    teacher.load_state_dict(
        {k.removeprefix(prefix): v for k, v in weights.items() if k.startswith(prefix)}
    )
    backbone = teacher.backbone
    backbone.config.pos_embed_shift = None
    backbone.config.pos_embed_jitter = None
    backbone.config.pos_embed_rescale = None
    for block in backbone.layer:
        block.drop_path = nn.Identity()
    return teacher.requires_grad_(False).eval().to(device)
