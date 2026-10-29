"""Render saved JEPA metrics without connecting to training or SwanLab."""

import argparse
import json
from pathlib import Path


def render(root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

    def line(axis, key, label):
        points = [(r["step"], r["metrics"][key]) for r in rows if key in r["metrics"]]
        if points:
            axis.plot(
                *zip(*points, strict=True), label=label, marker="." if len(points) < 8 else None
            )

    line(axes[0, 0], "train/prediction_loss", "train")
    line(axes[0, 0], "validation/all/prediction_loss", "fixed validation")
    axes[0, 0].set_title("Masked feature prediction (moving EMA target)")
    for group in ("Normal", "Low", "ExtremeLow", "Over", "ExtremeOver"):
        line(axes[0, 1], f"validation/{group}/prediction_loss", group)
    axes[0, 1].set_title("Fixed validation by lighting")
    line(axes[1, 0], "validation/all/prediction_loss", "correct context")
    line(axes[1, 0], "validation/all/swapped_context_loss", "swapped cameras")
    line(axes[1, 0], "validation/all/visible_mean_loss", "EMA visible mean (diagnostic)")
    axes[1, 0].set_title("Does prediction use image evidence?")
    line(axes[1, 1], "validation/all/target_spatial_std", "within image spatial std")
    line(axes[1, 1], "representation/validation_between_image_std", "between image std")
    axes[1, 1].set_title("Representation variation (not a pass/fail threshold)")
    for axis in axes.flat:
        axis.set_xlabel("Optimizer updates")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.savefig(root / "curves.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    render(args.run_root)


if __name__ == "__main__":
    main()
