"""Direct U2-to-U3 localization with frozen patch tokens and recorded-view annotations."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
from matplotlib.path import Path as Polygon
from torch.nn import functional as F

from owac.raw.all_bench_distillation import load_inventory, read_npz_member
from owac.raw.teleop_mcap import CAMERAS
from owac.training.raw_ssl import RawEncoder
from owac.training.raw_ssl_train import load_pretrained_encoder
from owac.training.teleop_probe import preview

matplotlib.use("Agg")


@torch.inference_mode()
def evaluate(cfg, run, labels, encoder=None, directory=None):
    directory = directory or run / "visual-probe"
    directory.mkdir(parents=True, exist_ok=True)
    annotations = json.loads(labels.read_text())["frames"]
    inventory, _ = load_inventory(cfg["data"]["inventory"])
    sources = {s.group_id: s for s in inventory if s.split == "validation"}
    device = torch.device("cuda")
    model = RawEncoder(cfg["model"]).to(device).eval()
    patch = cfg["model"]["patch_size"]
    scale = patch // 2
    samples = {}
    rows = []
    for key, checkpoint in [
        ("initial", cfg["training"]["initialize_encoder_from"]),
        ("adapted", encoder or run / "encoder.pt"),
    ]:
        step = load_pretrained_encoder(model, checkpoint)
        for annotated in annotations:
            source = sources[annotated["episode"]]
            frame = annotated["frame"]
            if not annotated["objects"]:
                continue
            raw = np.stack([read_npz_member(source.path, c + ".npy")[frame] for c in CAMERAS])
            h, w = raw.shape[1:]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, tokens = model(torch.as_tensor(raw, device=device).float())
            t = tokens.float()
            t = F.normalize(t - t.mean(1, keepdim=True), dim=-1)
            grid = (h // patch, w // patch)
            yy, xx = np.indices(grid)
            centers = np.stack(((xx + 0.5) * scale, (yy + 0.5) * scale), -1)
            for name, point, polygon in annotated["objects"]:
                x, y = point
                ix = int(y // scale) * grid[1] + int(x // scale)
                similarity = (t[1] @ t[0, ix]).cpu().numpy().reshape(grid)
                mask = Polygon(polygon).contains_points(centers.reshape(-1, 2)).reshape(grid)
                if not mask.any():
                    raise ValueError(f"Empty token region for {source.group_id}/{name}")
                order = (-similarity).ravel().argsort(kind="stable")
                best = centers.reshape(-1, 2)[order[0]]
                rank = int(np.flatnonzero(mask.ravel()[order])[0]) + 1
                row = dict(
                    model=key,
                    step=step,
                    episode=source.group_id,
                    frame=frame,
                    object=name,
                    rank=rank,
                    hit=rank == 1,
                    predicted_center=best.tolist(),
                    candidate_patches=int(mask.size),
                    object_patches=int(mask.sum()),
                )
                rows.append(row)
                identity = (source.group_id, frame, name)
                if identity not in samples:
                    sensor = cfg["model"]["measurement_encoder"]["sensor"]
                    samples[identity] = dict(
                        episode=source.group_id,
                        frame=frame,
                        object=name,
                        point=point,
                        polygon=polygon,
                        views=[
                            preview(v, sensor["black_level"], sensor["white_level"]) for v in raw
                        ],
                    )
                samples[identity][key] = (similarity, best, rank)
    objects = list(dict.fromkeys(ex["object"] for ex in samples.values()))
    result = {}
    for model in ("initial", "adapted"):
        subset = [r for r in rows if r["model"] == model]
        result[model] = {
            "hits": sum(r["hit"] for r in subset),
            "queries": len(subset),
            "mean_reciprocal_rank": float(np.mean([1 / r["rank"] for r in subset])),
            "per_object": {
                n: [
                    sum(r["hit"] for r in subset if r["object"] == n),
                    sum(r["object"] == n for r in subset),
                ]
                for n in objects
            },
        }
    (directory / "cross-camera-objects.json").write_text(
        json.dumps(dict(summary=result, queries=rows), indent=2)
    )
    # The first example of each object plus every erroneous query from either encoder.
    selected = [next(ex for ex in samples.values() if ex["object"] == name) for name in objects]
    selected += [
        ex
        for ex in samples.values()
        if all(ex is not previous for previous in selected)
        and (ex["initial"][2] != 1 or ex["adapted"][2] != 1)
    ]
    plot(selected, directory / "cross-camera-objects.png")
    print(json.dumps(result), flush=True)


def plot(samples, path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as Outline

    fig, axes = plt.subplots(len(samples), 4, figsize=(14, 2.5 * len(samples)), squeeze=False)
    for i, ex in enumerate(samples):
        ax = axes[i, 0]
        ax.imshow(ex["views"][0])
        ax.plot(*ex["point"], marker="+", color="#00ffff", ms=14, mew=2)
        ax.set_ylabel(ex["object"], fontsize=12)
        ax = axes[i, 1]
        ax.imshow(ex["views"][1])
        ax.add_patch(Outline(ex["polygon"], fill=False, edgecolor="#00ffff", linewidth=1.5))
        bounds = np.quantile(np.stack((ex["initial"][0], ex["adapted"][0])), [0.01, 0.99])
        for ax, key in zip(axes[i, 2:], ("initial", "adapted"), strict=True):
            similarity, best, rank = ex[key]
            h, w = ex["views"][1].shape[:2]
            ax.imshow(ex["views"][1])
            ax.imshow(
                similarity,
                cmap="inferno",
                extent=(-0.5, w - 0.5, h - 0.5, -0.5),
                alpha=0.65,
                vmin=bounds[0],
                vmax=bounds[1],
                interpolation="nearest",
            )
            ax.add_patch(Outline(ex["polygon"], fill=False, edgecolor="#00ffff", linewidth=1.2))
            ax.plot(*best, marker="x", color="white", ms=10, mew=2)
            ax.text(
                0.02,
                0.95,
                f"Rank {rank}",
                transform=ax.transAxes,
                va="top",
                color="white",
                fontsize=10,
                bbox=dict(facecolor="black", alpha=0.65, pad=1.5, edgecolor="none"),
            )
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
    for ax, title in zip(
        axes[0],
        ("U2 query", "U3 annotated target", "Simulation pretrained", "After real RAW training"),
        strict=True,
    ):
        ax.set_title(title, fontsize=13)
    fig.subplots_adjust(wspace=0.025, hspace=0.055, left=0.055, right=0.99, top=0.95, bottom=0.01)
    fig.savefig(path, dpi=132, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, help="Evaluate a saved intermediate EMA")
    parser.add_argument("--output", type=Path, help="Directory for the query scores and heatmap")
    args = parser.parse_args()
    evaluate(json.loads(args.config.read_text()), args.run, args.labels, args.encoder, args.output)
