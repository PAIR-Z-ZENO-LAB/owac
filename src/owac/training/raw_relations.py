"""Spatial geometry and soft temporal correspondence for RAW-only continuation."""

import math

import torch
from torch.nn import functional as F


def centered_tokens(tokens, masks=None):
    x = tokens.float()
    if masks is None:
        mean = x.mean(1, keepdim=True)
    else:
        visible = (~masks).unsqueeze(-1)
        mean = (x * visible).sum(1, keepdim=True) / visible.sum(1, keepdim=True)
    return F.normalize(x - mean, dim=-1)


def relation_loss(student, anchor, masks):
    """Preserve J3 pair relations, including its lower common cosine component."""
    s = F.normalize(student.float(), dim=-1)
    t = F.normalize(anchor.detach().float(), dim=-1)
    error = (s @ s.transpose(1, 2) - t @ t.transpose(1, 2)).square()
    visible = ~masks
    pairs = visible[:, :, None] & visible[:, None, :]
    pairs &= ~torch.eye(s.shape[1], dtype=torch.bool, device=s.device)[None]
    return error[pairs].mean()


def diversity_loss(tokens, masks):
    """VICReg-style moments on actual visible output tokens, not a throwaway head.

    Unit vectors scaled by sqrt(width) have unit marginal variance when isotropic.
    A sparse spatial sample bounds the covariance computation per rank.
    """
    grid = math.isqrt(tokens.shape[1])
    x = F.normalize(tokens.float(), dim=-1).reshape(-1, grid, grid, tokens.shape[-1])
    x = x[:, ::4, ::4].flatten(1, 2)
    sampled_mask = masks.reshape(-1, grid, grid)[:, ::4, ::4].flatten(1, 2)
    x = x[~sampled_mask] * math.sqrt(tokens.shape[-1])
    x = x - x.mean(0)
    covariance = x.T @ x / (len(x) - 1)
    variance = F.relu(1 - (covariance.diagonal() + 1e-4).sqrt()).mean()
    off = covariance - torch.diag_embed(covariance.diagonal())
    return variance, off.square().sum() / tokens.shape[-1]


@torch.no_grad()
def temporal_targets(current, previous, radius, temperature):
    """Soft local matching, weighted continuously by forward/backward agreement.

    No fixed-location pseudo-labels, RGB optical-flow model, or thresholded matches.
    These are noisy correspondences, not object masks or ground-truth motion.
    """
    a, b = centered_tokens(current), centered_tokens(previous)
    grid = math.isqrt(a.shape[1])
    yy, xx = torch.meshgrid(
        torch.arange(grid, device=a.device), torch.arange(grid, device=a.device), indexing="ij"
    )
    coordinates = torch.stack((yy.flatten(), xx.flatten()), -1)
    distance = (coordinates[:, None] - coordinates[None, :]).abs().amax(-1)
    scores = (a @ b.transpose(1, 2)) / temperature
    scores = scores.masked_fill(distance[None] > radius, -torch.inf)
    forward = scores.softmax(-1)
    backward = scores.transpose(1, 2).softmax(-1)
    confidence = (forward * backward.transpose(1, 2)).sum(-1)
    target = F.normalize(forward @ b, dim=-1)
    displacement = (forward * distance[None]).sum(-1).mean()
    return target, confidence, displacement


def temporal_loss(student, target, confidence, masks):
    s = centered_tokens(student, masks)
    error = 1 - (s * target.detach()).sum(-1)
    # Ambiguous matches contribute less, rather than being normalized back to full weight.
    return (error * confidence.detach())[~masks].mean()


def render_comparison(features, images, references, sources, cfg, directory):
    """Same RAW, same PCA basis and cosine scale; RGB is a display-only reference."""
    import matplotlib.pyplot as plt
    import numpy as np

    baseline = torch.from_numpy(np.load(cfg["evaluation"]["baseline_tokens"])["tokens"])
    both = torch.stack((baseline, features))
    flat = both.flatten(0, 2)
    mean = flat.mean(0)
    _, _, basis = torch.linalg.svd(flat - mean, full_matrices=False)
    projection = (both - mean) @ basis[:3].T
    low = torch.quantile(projection.flatten(0, 2), 0.01, dim=0)
    high = torch.quantile(projection.flatten(0, 2), 0.99, dim=0)
    colors = ((projection - low) / (high - low).clamp_min(torch.finfo(features.dtype).eps)).clamp(
        0, 1
    )
    normalized = F.normalize(both, dim=-1)
    paths = []
    for i in (0, 4, 10, 14):
        fig, axes = plt.subplots(2, 6, figsize=(18, 6))
        for row, label in enumerate(("J4 final", f"J5 {directory.name}")):
            axes[row, 0].imshow(images[i].permute(1, 2, 0).clamp(0, 1).pow(1 / 2.2))
            axes[row, 0].set_title(f"{label} | RAW")
            axes[row, 1].imshow(references[i])
            axes[row, 1].set_title("RGB: display only")
            axes[row, 2].imshow(colors[row, i].reshape(16, 16, 3))
            axes[row, 2].set_title("Joint PCA")
            for ax, (y, x) in zip(axes[row, 3:], ((4, 4), (8, 8), (12, 12)), strict=True):
                f = normalized[row, i]
                ax.imshow((f @ f[y * 16 + x]).reshape(16, 16), vmin=-1, vmax=1, cmap="viridis")
                ax.scatter([x], [y], marker="+", color="red")
                ax.set_title(f"Query ({y}, {x})")
            for ax in axes[row]:
                ax.axis("off")
        fig.suptitle(f"{sources[i].domain} | {sources[i].lighting} | raw cosine [-1, 1]")
        fig.tight_layout()
        path = directory / f"comparison-{i:02d}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(str(path))
    return paths
