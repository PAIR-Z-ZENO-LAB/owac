"""Fixed held-source PCA/query panels; no teacher or post-processing supervision."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from owac.raw.all_bench_distillation import DOMAIN_RAW_KEYS
from owac.training.raw_ssl_data import FrameReader, RawViews


@torch.no_grad()
def evaluate(encoder, sources, cfg, output, step, device):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    encoder.eval()
    reader = RawViews(sources, cfg)
    selected = []
    for domain in cfg["data"]["domains"]:
        for lighting in ("Normal", "Low", "ExtremeLow", "Over", "ExtremeOver"):
            group = [s for s in sources if s.domain == domain and s.lighting == lighting]
            selected.extend(group[:2])
    images, tokens, originals, references = [], [], [], []
    for source in selected:
        raw = reader.frame(source, source.num_frames // 2, DOMAIN_RAW_KEYS[source.domain][0])
        originals.append(raw)
        raw = F.interpolate(raw[None], size=(256, 256), mode="bilinear", align_corners=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, patch = encoder(raw.to(device))
        images.append(raw[0].cpu())
        tokens.append(patch[0].float().cpu())
    reader.close()
    features = torch.stack(tokens)
    flat = features.flatten(0, 1)
    # One PCA fit shared by ALL fixed validation images at this checkpoint.
    mean = flat.mean(0)
    _, singular, vectors = torch.linalg.svd(flat - mean, full_matrices=False)
    pca = (features - mean) @ vectors[:3].T
    low = torch.quantile(pca.flatten(0, 1), 0.01, dim=0)
    high = torch.quantile(pca.flatten(0, 1), 0.99, dim=0)
    colors = ((pca - low) / (high - low).clamp_min(torch.finfo(pca.dtype).eps)).clamp(0, 1)
    normalized = F.normalize(features, dim=-1)
    grid = 256 // cfg["model"]["patch_size"]
    queries = [(grid // 4, grid // 4), (grid // 2, grid // 2), (3 * grid // 4, 3 * grid // 4)]
    directory = Path(output) / "visuals" / f"{step:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, source in enumerate(selected):
        show_rgb = cfg.get("evaluation", {}).get("rgb_reference", False)
        fig, axes = plt.subplots(1, 6 if show_rgb else 5, figsize=(18 if show_rgb else 15, 3))
        if show_rgb:
            rgb_reader = FrameReader(
                str(source.path).replace("/raw/", "/rgb/"),
                "agentview_rgb_uint8" if source.domain == "libero" else "head_camera_rgb_uint8",
            )
            rgb = rgb_reader.frame(source.num_frames // 2)
            rgb_reader.close()
            rgb = (
                F.interpolate(
                    torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float(),
                    size=(256, 256),
                    mode="bilinear",
                    align_corners=False,
                )[0].permute(1, 2, 0)
                / 255
            )
            references.append(rgb)
            axes[1].imshow(rgb)
            axes[1].set_title("Paired RGB: display only")
            axes[1].axis("off")
            axes = [axes[0], *axes[2:]]
        axes[0].imshow(images[i].permute(1, 2, 0).clamp(0, 1).pow(1 / 2.2))
        axes[0].set_title("RAW preview (gamma only)")
        axes[1].imshow(colors[i].reshape(grid, grid, 3))
        axes[1].set_title("Tokens: fixed-panel PCA")
        for ax, (y, x) in zip(axes[2:], queries, strict=True):
            similarity = normalized[i] @ normalized[i, y * grid + x]
            ax.imshow(similarity.reshape(grid, grid), vmin=-1, vmax=1, cmap="viridis")
            ax.scatter([x], [y], marker="+", color="red")
            ax.set_title(f"Query ({y}, {x})")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"RAW SSL step {step} | {source.domain} | {source.lighting} | RAW only")
        fig.tight_layout()
        path = directory / f"{i:02d}-{source.domain}-{source.lighting}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        paths.append(str(path))
    pooled = F.normalize(features.mean(1), dim=-1)
    metrics = {
        "eval/patch_spatial_std": float(normalized.std(1).mean()),
        "eval/between_image_cosine": float(
            (pooled @ pooled.T).fill_diagonal_(0).sum() / (len(pooled) * (len(pooled) - 1))
        ),
        "eval/images": len(selected),
    }
    energy = singular.square() / singular.square().sum()
    metrics["eval/effective_rank"] = float(
        torch.exp(-(energy * energy.clamp_min(torch.finfo(energy.dtype).tiny).log()).sum())
    )
    grams = normalized @ normalized.transpose(1, 2)
    off = ~torch.eye(features.shape[1], dtype=torch.bool)
    metrics["eval/patch_offdiag_cosine"] = float(grams[:, off].mean())
    for domain in cfg["data"]["domains"]:
        for lighting in ("Normal", "Low", "ExtremeLow", "Over", "ExtremeOver"):
            ids = [
                i for i, s in enumerate(selected) if s.domain == domain and s.lighting == lighting
            ]
            metrics[f"eval/{domain}/{lighting}/patch_cosine"] = float(grams[ids][:, off].mean())
    if "exposure" in cfg:
        from owac.training.raw_exposure import evaluate_exposure

        extra, pair_paths = evaluate_exposure(
            encoder, originals, selected, features, vectors[:3], mean, cfg, directory, device
        )
        metrics.update(extra)
        paths.extend(pair_paths)
    if cfg.get("evaluation", {}).get("baseline_tokens"):
        from owac.training.raw_relations import render_comparison

        paths.extend(render_comparison(features, images, references, selected, cfg, directory))
    (directory / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez(
        directory / "tokens.npz",
        tokens=features.numpy(),
        pca_basis=vectors[:3].numpy(),
        pca_mean=mean.numpy(),
    )
    return metrics, paths
