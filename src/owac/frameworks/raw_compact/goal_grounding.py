"""Training-only image-plane supervision for demonstrated future motion points."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class SpatialGoalHead(nn.Module):
    """Locate each arm's next waypoint from current patches and task language.

    No proprioception, command history, future image or target coordinate enters
    the prediction. This auxiliary head is never called by policy sampling.
    """

    def __init__(self, visual_width, language_width, width, arms):
        super().__init__()
        self.key = nn.Sequential(nn.LayerNorm(visual_width), nn.Linear(visual_width, width))
        self.language = nn.Linear(language_width, width)
        self.arm_queries = nn.Parameter(torch.randn(arms, width) / math.sqrt(width))
        self.scale = width**-0.5

    def forward(self, patches, language):
        query = self.language(language)[:, None] + self.arm_queries[None]
        return torch.einsum("bad,bpd->bap", query, self.key(patches)) * self.scale


def spatial_goal_loss(logits, uv, projectable, grid_size, *, group_size=None):
    """Cross entropy against bilinear grid weights, with no Gaussian bandwidth.

    uv is in pixel-centre normalized image coordinates: (pixel +0.5)/image_size.
    Validity comes from positive camera depth, image bounds and existence of a
    future demonstration label. It does not test occlusion or model confidence.
    """
    height, width = grid_size
    projectable = projectable.bool()
    coordinate = torch.where(projectable[..., None], uv.float(), 0)
    x = (coordinate[..., 0] * width - 0.5).clamp(0, width - 1)
    y = (coordinate[..., 1] * height - 0.5).clamp(0, height - 1)
    x0, y0 = x.floor().long(), y.floor().long()
    x1, y1 = (x0 + 1).clamp_max(width - 1), (y0 + 1).clamp_max(height - 1)
    dx, dy = x - x0, y - y0
    indices = torch.stack((y0 * width + x0, y0 * width + x1, y1 * width + x0, y1 * width + x1), -1)
    weights = torch.stack(((1 - dx) * (1 - dy), dx * (1 - dy), (1 - dx) * dy, dx * dy), -1)
    log_probability = F.log_softmax(logits.float(), dim=-1)
    terms = -(log_probability.gather(-1, indices) * weights).sum(-1)
    count = projectable.sum()
    if group_size is None:
        loss = (terms * projectable).sum() / count.clamp_min(1)
    else:
        # An empty logical group contributes zero, just as a separate old
        # microbatch did; other groups keep their own projectable denominator.
        weighted = (terms * projectable).reshape(-1, group_size, terms.shape[-1])
        counts = projectable.reshape_as(weighted).sum(dim=(1, 2))
        loss = (weighted.sum(dim=(1, 2)) / counts.clamp_min(1)).mean()
    return loss, count
