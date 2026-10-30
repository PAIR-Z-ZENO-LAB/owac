"""RAW-only exposure pairs, preserving geometry and accounting for existing noise."""

import torch
from torch.nn import functional as F


def lower_exposure(raw, stops, noise, generator):
    """Approximate another exposure of an already noisy linear measurement.

    For Var(x)=shot*s+read, scaling by a already retains a² Var(x).
    Add only a(1-a)*shot*x + (1-a²)*read, before spatial resampling.
    Clipping and quantization make this an augmentation, not a clean sensor simulator.
    """
    gain = 2.0 ** (-stops)
    variance = gain * (1 - gain) * noise["shot"] * raw.clamp_min(0)
    variance = variance + (1 - gain * gain) * noise["read_var"]
    result = gain * raw + variance.sqrt() * torch.randn(
        raw.shape, generator=generator, dtype=raw.dtype, device=raw.device
    )
    result = result.clamp(0, noise["white_level"])
    if noise["levels"]:
        result = (result * noise["levels"]).round() / noise["levels"]
    return result


def dense_consistency(student, target, masks):
    """Align actual output tokens on visible patches; the EMA target has no gradient."""
    similarity = F.cosine_similarity(student.float(), target.detach().float(), dim=-1)
    return ((1 - similarity) * (~masks)).sum() / (~masks).sum()


@torch.no_grad()
def evaluate_exposure(encoder, originals, sources, features, basis, mean, cfg, directory, device):
    """Fixed synthetic pairs plus shifted-crop matching to expose position shortcuts."""
    import matplotlib.pyplot as plt

    metrics, paths = {}, []
    patch = cfg["model"]["patch_size"]
    grid = 256 // patch
    yy, xx = torch.meshgrid(torch.arange(grid - 1), torch.arange(grid - 1), indexing="ij")
    expected = ((yy + 1) * grid + xx + 1).flatten().to(device)
    for i, source in enumerate(sources):
        if source.lighting != "Normal":
            continue
        reference = F.normalize(features[i].to(device), dim=-1)
        rows = []
        for stops in (0, 2, 4, 6):
            dark = (
                lower_exposure(
                    originals[i],
                    stops,
                    cfg["exposure"]["noise"][source.domain],
                    torch.Generator().manual_seed(731 + i),
                )
                if stops
                else originals[i]
            )
            dark = F.interpolate(dark[None], size=(256, 256), mode="bilinear", align_corners=False)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, tokens = encoder(dark.to(device))
                _, shifted = encoder(dark[:, :, patch:, patch:].to(device))
            normalized = F.normalize(tokens[0].float(), dim=-1)
            matching = F.normalize(shifted[0].float(), dim=-1) @ reference.T
            prefix = f"exposure/{source.domain}/minus_{stops}ev"
            metrics.setdefault(prefix + "/cosine", []).append(
                float((normalized * reference).sum(-1).mean())
            )
            metrics.setdefault(prefix + "/shifted_top1", []).append(
                float((matching.argmax(-1) == expected).float().mean())
            )
            projection = (tokens[0].float().cpu() - mean) @ basis.T
            rows.append((dark[0], projection, normalized.cpu()))
        # One reference-derived scale, not independently stretched dark images.
        projected = (features[i] - mean) @ basis.T
        low, high = projected.amin(0), projected.amax(0)
        fig, axes = plt.subplots(4, 5, figsize=(15, 12))
        for row, (stops, (raw, projection, normalized)) in enumerate(
            zip((0, 2, 4, 6), rows, strict=True)
        ):
            axes[row, 0].imshow(raw.permute(1, 2, 0).clamp(0, 1).pow(1 / 2.2))
            axes[row, 0].set_title(f"Synthetic -{stops} EV RAW")
            axes[row, 1].imshow(
                ((projection - low) / (high - low).clamp_min(1e-8))
                .clamp(0, 1)
                .reshape(grid, grid, 3)
            )
            axes[row, 1].set_title("Same reference PCA")
            for ax, (y, x) in zip(axes[row, 2:], ((4, 4), (8, 8), (12, 12)), strict=True):
                ax.imshow(
                    (normalized @ normalized[y * grid + x]).reshape(grid, grid),
                    vmin=-1,
                    vmax=1,
                    cmap="viridis",
                )
                ax.scatter([x], [y], marker="+", color="red")
                ax.set_title(f"Query ({y}, {x})")
            for ax in axes[row]:
                ax.axis("off")
        fig.suptitle(f"RAW exposure probe | {source.domain} | source {i} | {directory.name}")
        fig.tight_layout()
        path = directory / f"exposure-{i:02d}-{source.domain}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        paths.append(str(path))
    return {key: sum(values) / len(values) for key, values in metrics.items()}, paths
