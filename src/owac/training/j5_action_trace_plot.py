"""Render replay observations, fixed-basis token PCA, query maps and real attention."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    files = sorted((root / "features").glob("*.npz"))
    features = {p.stem: dict(np.load(p)) for p in files}
    inputs = {p.stem: dict(np.load(root / "inputs" / p.name)) for p in files}
    colors, pca_info = {}, {}
    torch.set_num_threads(8)
    for name in ["raw_tokens", "projected_tokens", "contextual_tokens"]:
        x = torch.from_numpy(
            np.concatenate(
                [features[k][name].reshape(-1, features[k][name].shape[-1]) for k in features]
            )
        )
        torch.manual_seed(0)
        mean = x.mean(0)
        _, singular, basis = torch.pca_lowrank(x, q=12, center=True, niter=4)
        scores = ((x - mean) @ basis[:, :3]).numpy()
        lo, hi = np.percentile(scores, [1, 99], axis=0)
        rgb = np.clip((scores - lo) / (hi - lo), 0, 1)
        colors[name] = {
            k: rgb[i * 512 : (i + 1) * 512].reshape(2, 16, 16, 3) for i, k in enumerate(features)
        }
        pca_info[name] = {
            "variance_fraction_3d": float(singular[:3].square().sum() / (x - mean).square().sum()),
            "normalization": "fixed 1st/99th percentiles over all lights/frames/views",
        }
    (root / "pca.json").write_text(json.dumps(pca_info, indent=2))
    attention_max = max(
        float(v["attention"][:, :, :, :512].mean((1, 2)).max()) for v in features.values()
    )
    titles = [
        "Rendered scene (display)",
        "RAW preview (gamma only)",
        "Frozen J5 tokens: PCA",
        "Bridge tokens: PCA",
        "Language-context tokens: PCA",
        "Action -> vision attention (L18)",
    ]
    for light in ["Normal", "ExtremeLow"]:
        fig, axes = plt.subplots(3, 6, figsize=(18, 9), constrained_layout=True)
        for row, step in enumerate([0, 50, 200]):
            key = f"{light}-{step:03d}"
            data, feat = inputs[key], features[key]
            images = [data["display"][0], (data["raw"][0] / 255) ** (1 / 2.2)]
            images += [
                colors[n][key][0] for n in ["raw_tokens", "projected_tokens", "contextual_tokens"]
            ]
            for col, im in enumerate(images):
                axes[row, col].imshow(im, interpolation="nearest")
            ax = axes[row, 5]
            ax.imshow(data["display"][0])
            att = feat["attention"][2, :, :, :256].mean((0, 1)).reshape(16, 16)
            heat = ax.imshow(
                att,
                extent=(-0.5, 255.5, 255.5, -0.5),
                cmap="magma",
                alpha=0.68,
                norm=LogNorm(vmin=1e-5, vmax=attention_max),
                interpolation="nearest",
            )
            mass = feat["attention"][2, :, :, :256].mean((0, 1)).sum()
            ax.set_xlabel(f"external-view mass = {mass:.1%}")
            for col in range(6):
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
                if row == 0:
                    axes[row, col].set_title(titles[col], fontsize=10)
            axes[row, 0].set_ylabel(f"Control step {step}", fontsize=12)
        fig.colorbar(
            heat,
            ax=axes[:, 5],
            shrink=0.65,
            label="Mean probability / visual token (fixed log scale)",
        )
        fig.suptitle(
            f"{light} | expert step 5000 | recorded failed trajectory\n"
            "PCA basis fixed over time per representation; attention averages 8 heads, "
            "10 denoising steps, first 5 action positions",
            fontsize=13,
        )
        fig.savefig(root / f"{light}-timeline.png", dpi=140)
        plt.close(fig)
    # Expose layer dependence instead of interpreting one attention average as a segmentation.
    fig, axes = plt.subplots(2, 5, figsize=(16, 7), constrained_layout=True)
    key = "Normal-000"
    feat = features[key]
    data = inputs[key]
    for view in range(2):
        axes[view, 0].imshow(data["display"][view])
        axes[view, 1].imshow(colors["raw_tokens"][key][view], interpolation="nearest")
        for col, idx in enumerate([0, 1, 2], start=2):
            att = (
                feat["attention"][idx, :, :, view * 256 : (view + 1) * 256]
                .mean((0, 1))
                .reshape(16, 16)
            )
            axes[view, col].imshow(data["display"][view])
            axes[view, col].imshow(
                att,
                extent=(-0.5, 255.5, 255.5, -0.5),
                cmap="magma",
                alpha=0.7,
                norm=LogNorm(vmin=1e-5, vmax=attention_max),
                interpolation="nearest",
            )
            axes[view, col].set_xlabel(f"view mass = {att.sum():.1%}")
        for col, title in enumerate(
            [
                "Scene",
                "J5 tokens PCA",
                "Action attention L1",
                "Action attention L9",
                "Action attention L18",
            ]
        ):
            axes[view, col].set_title(title)
            axes[view, col].set_xticks([])
            axes[view, col].set_yticks([])
        axes[view, 0].set_ylabel(["External camera", "Wrist camera"][view])
    fig.suptitle(
        "Normal, control step 0 | actual softmax attention; averages are not causal explanations"
    )
    fig.savefig(root / "attention-layers.png", dpi=140)
    plt.close(fig)
    # Same initial target patch as query across time, no object-label supervision.
    fig, axes = plt.subplots(3, 4, figsize=(12, 9), constrained_layout=True)
    for row, step in enumerate([0, 50, 200]):
        key = f"Normal-{step:03d}"
        axes[row, 0].imshow(inputs[key]["display"][0])
        axes[row, 0].plot(72, 152, "r+")
        for col, name in enumerate(
            ["raw_tokens", "projected_tokens", "contextual_tokens"], start=1
        ):
            q = features["Normal-000"][name][0, 9 * 16 + 4]
            x = features[key][name][0]
            similarity = (x @ q) / (np.linalg.norm(x, axis=-1) * np.linalg.norm(q))
            heat = axes[row, col].imshow(
                similarity.reshape(16, 16), vmin=-1, vmax=1, cmap="viridis", interpolation="nearest"
            )
        for col, title in enumerate(
            [
                "Scene (initial query location)",
                "J5 target-anchor cosine",
                "Bridge target-anchor cosine",
                "Context target-anchor cosine",
            ]
        ):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(title, fontsize=10)
        axes[row, 0].set_ylabel(f"Control step {step}")
    fig.colorbar(heat, ax=axes[:, 1:], shrink=0.6, label="Cosine similarity [-1, 1]")
    fig.suptitle("Normal | fixed query token from step 0 at grid (row 9, col 4), near target bowl")
    fig.savefig(root / "query-timeline.png", dpi=140)
    plt.close(fig)
    # Stream mass per selected layer and control instant.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    stream_colors = ["#2878B5", "#F8AC8C", "#9AC9DB", "#C82423"]
    for ax, light in zip(axes, ["Normal", "ExtremeLow"], strict=True):
        ks = [k for k in features if k.startswith(light + "-")]
        probs = np.stack([features[k]["attention"][2].mean((0, 1)) for k in ks])
        masses = np.stack(
            [
                probs[:, :256].sum(1),
                probs[:, 256:512].sum(1),
                probs[:, 512:-10].sum(1),
                probs[:, -10:].sum(1),
            ]
        )
        ax.stackplot(
            [int(k.split("-")[-1]) for k in ks],
            masses,
            labels=["External", "Wrist", "Text", "Action"],
            colors=stream_colors,
        )
        ax.set_ylim(0, 1)
        ax.set_title(light + " | last layer")
        ax.set_xlabel("Control step")
        ax.set_ylabel("Attention mass")
    axes[1].legend(loc="upper right")
    fig.savefig(root / "attention-mass.png", dpi=140)
    plt.close(fig)
    baseline = dict(np.load(root / "baseline-features" / "Normal-000.npz"))
    raw_first = features["Normal-000"]
    fig, axes = plt.subplots(2, 4, figsize=(13, 7), constrained_layout=True)
    for row, (label, feature) in enumerate(
        [("RAW / failed", raw_first), ("RGB / succeeded", baseline)]
    ):
        axes[row, 0].imshow(inputs["Normal-000"]["display"][0])
        axes[row, 0].plot(72, 152, "r+")
        axes[row, 0].set_ylabel(label)
        for col, name in enumerate(
            ["raw_tokens", "projected_tokens", "contextual_tokens"], start=1
        ):
            x = feature[name][0]
            q = x[9 * 16 + 4]
            similarity = (x @ q) / (np.linalg.norm(x, axis=-1) * np.linalg.norm(q))
            heat = axes[row, col].imshow(
                similarity.reshape(16, 16), vmin=-1, vmax=1, cmap="viridis"
            )
        for col, title in enumerate(
            [
                "Same initial scene / query",
                "Vision encoder query",
                "Projector query",
                "Language-context query",
            ]
        ):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(title)
    fig.colorbar(heat, ax=axes[:, 1:], shrink=0.65, label="Within-model cosine [-1, 1]")
    fig.suptitle("Same scene and target patch | RAW vs successful official RGB reference")
    fig.savefig(root / "raw-rgb-query.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(13, 7), constrained_layout=True)
    vmax = max(
        float(x["attention"][:, :, :, :256].mean((1, 2)).max()) for x in [raw_first, baseline]
    )
    for row, (label, feature) in enumerate(
        [("RAW / failed", raw_first), ("RGB / succeeded", baseline)]
    ):
        axes[row, 0].imshow(inputs["Normal-000"]["display"][0])
        axes[row, 0].set_ylabel(label)
        for col, idx in enumerate([0, 1, 2], start=1):
            att = feature["attention"][idx, :, :, :256].mean((0, 1)).reshape(16, 16)
            axes[row, col].imshow(inputs["Normal-000"]["display"][0])
            heat = axes[row, col].imshow(
                att,
                extent=(-0.5, 255.5, 255.5, -0.5),
                cmap="magma",
                alpha=0.7,
                norm=LogNorm(vmin=1e-5, vmax=vmax),
                interpolation="nearest",
            )
            axes[row, col].set_xlabel(f"external-view mass = {att.sum():.1%}")
        for col, title in enumerate(
            ["Same scene", "Action attention L1", "Action attention L9", "Action attention L18"]
        ):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(title)
    fig.colorbar(
        heat, ax=axes[:, 1:], shrink=0.65, label="Actual probability / token (fixed log scale)"
    )
    fig.suptitle("Same initial scene | 8 heads, 10 denoising iterations, first 5 action positions")
    fig.savefig(root / "raw-rgb-attention.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
    for row, (label, feature) in enumerate([("RAW", raw_first), ("RGB", baseline)]):
        x = feature["contextual_tokens"][0]
        for col, center in enumerate([False, True]):
            y = x - x.mean(0) if center else x
            q = y[9 * 16 + 4]
            similarity = (y @ q) / (np.linalg.norm(y, axis=-1) * np.linalg.norm(q))
            heat = axes[row, col].imshow(
                similarity.reshape(16, 16), vmin=-1, vmax=1, cmap="viridis"
            )
            axes[row, col].axis("off")
            axes[row, col].set_title(
                label + (" / scene-mean removed" if center else " / original cosine")
            )
    fig.colorbar(heat, ax=axes.ravel().tolist(), shrink=0.7)
    fig.savefig(root / "context-centered-query.png", dpi=140)
    plt.close(fig)
    print(json.dumps(pca_info), flush=True)


if __name__ == "__main__":
    main()
