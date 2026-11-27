"""Estimate a low-signal RAW augmentation from the dark training recordings only."""

import argparse
import json
from pathlib import Path

import numpy as np

from owac.raw.all_bench_distillation import load_inventory
from owac.training.teleop_policy_data import raw_views
from owac.training.teleop_probe import brightness_pairs


def fit(sources):
    wall = []
    per_episode = []
    for src in sources:
        views = raw_views(src)
        wall.extend(brightness_pairs(views[0], 1)[0].tolist())
        with np.load(src.path) as z:
            motion = np.linalg.norm(np.diff(z["state"][:, :7], axis=0), axis=1)
        # The quietest consecutive observations limit scene motion in the wrist view.
        quiet = np.flatnonzero(motion <= np.quantile(motion, 0.1))
        sigmas = []
        for view in views:
            channels = []
            for y, x in ((0, 0), (0, 1), (1, 0), (1, 1)):
                a = view[quiet, y::16, x::16].astype(np.float32)
                b = view[quiet + 1, y::16, x::16].astype(np.float32)
                # Estimate noise near the dark pedestal. Differencing cancels fixed offsets.
                low = a + b <= np.median(a + b)
                lo, hi = np.quantile((b - a)[low], [0.15865, 0.84135])
                channels.append(float((hi - lo) / (2 * np.sqrt(2))))
            sigmas.append(channels)
        per_episode.append(sigmas)
    levels, counts = np.unique(wall, return_counts=True)
    return {
        "episodes": [s.group_id for s in sources],
        "wall_levels_dn": levels.tolist(),
        "wall_counts": counts.tolist(),
        "noise_sigma_dn": np.median(per_episode, axis=0).tolist(),
        "noise_sigma_dn_by_episode": per_episode,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    dark = [
        s for s in load_inventory(args.inventory)[0] if s.suite == "fruit03" and s.split == "train"
    ]
    args.output.write_text(json.dumps(fit(dark), indent=2) + "\n")
