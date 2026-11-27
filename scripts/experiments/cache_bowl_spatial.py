"""Fit projection axes on training RAW only and cache spatial OWAC features for local refitting."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from owac.frameworks.raw_compact.spatial_regression import SpatialActionRegression
from owac.raw.all_bench_distillation import load_inventory
from owac.training.teleop_policy_data import raw_views
from owac.training.teleop_recovery import load_export


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment", type=Path)
    p.add_argument("parent", type=Path)
    args = p.parse_args()
    torch.set_num_threads(2)
    model, _ = load_export(args.parent)
    dest = args.experiment / "spatial"
    dest.mkdir(exist_ok=True)
    episodes, _ = load_inventory(args.experiment / "aligned/inventory.json")

    def patches(views, ix):
        raw = torch.from_numpy(np.stack([v[ix] for v in views], axis=1).astype("f4")).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model.encode_raw(raw)[1]

    project = dest / "projection.pt"
    if not project.exists():
        width = model.cfg["model"]["raw_encoder"]["width"]
        covariance = torch.zeros(width, width, device="cuda", dtype=torch.float64)
        total = torch.zeros(width, device="cuda", dtype=torch.float64)
        count = 0
        for e in episodes:
            if e.split != "train":
                continue
            views = raw_views(e)
            ix = np.linspace(0, e.num_frames - 1, 8).astype(int)
            x = patches(views, ix)[:, ::4].flatten(0, 1).double()
            covariance += x.T @ x
            total += x.sum(0)
            count += len(x)
        mean = total / count
        vals, vecs = torch.linalg.eigh(covariance / count - torch.outer(mean, mean))
        w = (vecs[:, -32:] / vals[-32:].sqrt()).T
        torch.save(dict(weight=w.float().cpu(), bias=(-w @ mean).float().cpu()), project)
        (dest / "fit.json").write_text(
            json.dumps(
                dict(
                    episodes=[e.group_id for e in episodes if e.split == "train"],
                    patches=count,
                    explained_fraction=float(vals[-32:].sum() / vals.sum()),
                )
            )
        )
    model.compress = nn.Linear(model.cfg["model"]["raw_encoder"]["width"], 32).cuda()
    model.compress.load_state_dict(torch.load(project, weights_only=True))
    for e in episodes:
        path = dest / (e.group_id + ".npy")
        if path.exists():
            continue
        path.parent.mkdir(exist_ok=True)
        temp = path.with_suffix(".partial.npy")
        views = raw_views(e)
        for i in range(0, e.num_frames, 12):
            tokens = patches(views, slice(i, i + 12))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                spatial = SpatialActionRegression.spatial_map(model, tokens, tokens.shape[0] // 2)
            if i == 0:
                bank = np.lib.format.open_memmap(
                    temp, mode="w+", dtype="int16", shape=(e.num_frames, *spatial.shape[1:])
                )
            bank[i : i + 12] = spatial.to(torch.bfloat16).view(torch.int16).cpu().numpy()
        bank.flush()
        del bank
        temp.replace(path)
        print(e.group_id, e.num_frames, flush=True)


if __name__ == "__main__":
    main()
