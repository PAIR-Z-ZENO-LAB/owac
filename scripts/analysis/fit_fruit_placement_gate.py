"""Learn when a grasped fruit is being carried, with labels from demonstration commands."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from owac.frameworks.raw_compact.fruit_placement import PlacementGate, placement_descriptor
from owac.raw.all_bench_distillation import load_inventory
from owac.training.fruit_placement import placement_events
from owac.training.teleop_policy_data import cache_paths


def descriptors(src, cfg, grid, complete=False):
    path, info = cache_paths(cfg, src)
    sig = json.loads(info.read_text())
    if (sig["encoder"], sig["dataset_revision"], sig["episode"]) != (
        cfg["assets"]["visual_sha256"],
        cfg["data"]["dataset_revision"],
        src.group_id,
    ):
        raise ValueError(f"Feature cache differs for {src.group_id}")
    feature = np.load(path, mmap_mode="r")
    stats = cfg["statistics"]
    with np.load(src.path) as z:
        q = (z["state"] - np.array(stats["state_mean"])) / (np.array(stats["state_std"]) + 1e-6)
    parts = []
    for i in range(0, len(q), 32):
        raw = (
            torch.from_numpy(feature[i : i + 32].view(np.int16).copy()).view(torch.bfloat16).cuda()
        )
        parts.append(placement_descriptor(raw, torch.tensor(q[i : i + 32], device="cuda"), grid))
    e = placement_events(src)
    start = e.grasp_complete if complete else e.close
    phase = np.select(
        [e.time < start, e.time < e.release, e.time <= e.opening_end], [0, 1, 2], default=3
    )
    return torch.cat(parts), phase


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--completed-grasp", action="store_true")
    p.add_argument("--save-descriptors", type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20261120)
    torch.set_num_threads(2)
    cfg = torch.load(args.checkpoint, mmap=True, map_location="cpu", weights_only=False)["config"]
    enc = cfg["model"]["raw_encoder"]
    grid = tuple(x // enc["patch_size"] for x in cfg["observation"]["raw_shape"])
    gate = PlacementGate(enc["width"]).cuda()
    data, groups, start = [], [], 0
    for src in load_inventory(cfg["data"]["inventory"])[0]:
        x, stage = descriptors(src, cfg, grid, args.completed_grasp)
        if args.save_descriptors:
            path = args.save_descriptors / (src.group_id + ".pt")
            path.parent.mkdir(exist_ok=True, parents=True)
            torch.save(x.cpu(), path)
        data.append(x)
        groups.append((src, [start + np.flatnonzero(stage == i) for i in range(4)]))
        start += len(x)
    x = torch.cat(data)
    train = [pools for s, pools in groups if s.split == "train"]
    valid = [(s.group_id, pools) for s, pools in groups if s.split == "validation"]
    train_ix = np.concatenate([rows for pools in train for rows in pools])
    gate.mean.copy_(x[train_ix].mean(0))
    gate.scale.copy_(x[train_ix].std(0).clamp_min(1e-6))
    opt = torch.optim.AdamW(gate.parameters(), lr=0.0004)
    rng = np.random.default_rng(20261120)
    best = float("inf")
    for step in range(1, 1001):
        # Equal weights per trajectory and per stage, including the return after release.
        ids = rng.integers(len(train), size=128)
        phases = rng.integers(4, size=128)
        ix = [rng.choice(train[i][phase]) for i, phase in zip(ids, phases, strict=True)]
        target = torch.tensor(np.isin(phases, [1, 2]), device="cuda").long()
        loss = F.cross_entropy(gate(x[ix]), target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 100:
            continue
        with torch.no_grad():
            predicted = gate(x).argmax(-1)
            by_episode = {
                eid: [
                    (predicted[indices] == int(i in (1, 2))).float().mean().item()
                    for i, indices in enumerate(pools)
                ]
                for eid, pools in valid
            }
        error = 1 - np.mean(list(by_episode.values()))
        if error < best:
            best = error
            torch.save(gate.state_dict(), args.output / "gate.pt")
            (args.output / "gate.json").write_text(
                json.dumps(
                    dict(
                        step=step,
                        validation_stage_balanced_accuracy=1 - error,
                        per_stage_accuracy=by_episode,
                        stage_order=["before_close", "carrying", "opening_stroke", "after_opening"],
                        training_episodes=[s.group_id for s, _ in groups if s.split == "train"],
                    ),
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
