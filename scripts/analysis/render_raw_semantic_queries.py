"""Render fixed-position RAW/DINO query affinities with a common cosine scale."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads((args.root / "manifest.json").read_text())
    data = np.load(args.root / "affinities.npz")
    for i, sample in enumerate(manifest["samples"]):
        fig, axes = plt.subplots(2, 4, figsize=(12, 6), dpi=140, layout="constrained")
        axes[0, 0].imshow(np.clip(data[f"{i}_raw_preview"], 0, 1))
        axes[0, 0].set_title("RAW display (gamma only)")
        axes[1, 0].imshow(np.clip(data[f"{i}_rgb"], 0, 1))
        axes[1, 0].set_title("Paired normal-light RGB")
        for row, name in enumerate(("raw", "dino")):
            for column, (y, x) in enumerate(manifest["queries"], 1):
                grid = data[f"{i}_{name}_affinity"][y * 16 + x].reshape(16, 16)
                image = axes[row, column].imshow(grid, vmin=-1, vmax=1, cmap="viridis")
                axes[row, column].plot(x, y, "+", color="red", ms=12, mew=2)
                axes[row, column].set_title(f"{name.upper()} query ({x}, {y})")
        for ax in axes.flat:
            ax.set_axis_off()
        fig.colorbar(image, ax=axes[:, 1:], shrink=0.8, label="cosine similarity [-1, 1]")
        fig.suptitle(f"J2S step {manifest['step']} | {sample['lighting']} | fixed queries")
        fig.savefig(args.root / f"{i}_{sample['lighting']}.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
