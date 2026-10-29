"""Dense RAW refinement: fixed spatial anchor and known crop correspondence.

Gram anchoring is adapted from DINOv3; the anchor here is OWAC J1 block 9,
not DINOv3. Crop correspondence is a local geometric objective, not semantics.
"""

import copy

import torch
from torch.nn import functional as F


def fixed_spatial_anchor(encoder, depth):
    anchor = copy.deepcopy(encoder)
    anchor.vision.encoder.layers = anchor.vision.encoder.layers[:depth]
    anchor.refinement_layer = None
    del anchor.refinement_scale
    anchor.vision.post_layernorm = torch.nn.Identity()
    return anchor.requires_grad_(False).eval()


def overlapping_crops(native, rng, *, size=224, patch=14):
    """Use one transform for the complete causal window and both cameras."""
    height, width = native.shape[-2:]
    origins = [
        (y, x)
        for y in range(0, height - size + 1, patch)
        for x in range(0, width - size + 1, patch)
    ]
    a, b = rng.sample(origins, 2)
    views = tuple(native[..., y : y + size, x : x + size] for y, x in (a, b))
    ia, ib = crop_correspondence(a, b, size=size, patch=patch, device=native.device)
    return views, ia, ib


def crop_correspondence(a, b, *, size=224, patch=14, device=None):
    """Exact patch-center mapping; omit stem boundary patches in both views."""
    grid = size // patch
    yy, xx = torch.meshgrid(
        torch.arange(grid, device=device), torch.arange(grid, device=device), indexing="ij"
    )
    by, bx = yy + (a[0] - b[0]) // patch, xx + (a[1] - b[1]) // patch
    valid = (yy > 0) & (yy < grid - 1) & (xx > 0) & (xx < grid - 1)
    valid &= (by > 0) & (by < grid - 1) & (bx > 0) & (bx < grid - 1)
    return (yy * grid + xx)[valid], (by * grid + bx)[valid]


def gram_loss(features, reference):
    features = F.normalize(features.float(), dim=-1)
    reference = F.normalize(reference.detach().float(), dim=-1)
    return F.mse_loss(
        features @ features.transpose(-1, -2), reference @ reference.transpose(-1, -2)
    )


def dense_objective(a, b, anchor_a, anchor_b, ia, ib, *, temperature=0.1):
    """Symmetric within-view negatives, exact physical positives, fixed Gram target."""
    qa, qb = F.normalize(a[:, ia].float(), dim=-1), F.normalize(b[:, ib].float(), dim=-1)
    logits = qa @ qb.transpose(-1, -2) / temperature
    labels = torch.arange(len(ia), device=a.device).expand(a.shape[0], -1)
    correspondence = (
        F.cross_entropy(logits.transpose(1, 2), labels) + F.cross_entropy(logits, labels)
    ) / 2
    gram = (gram_loss(a, anchor_a) + gram_loss(b, anchor_b)) / 2
    with torch.no_grad():
        matches = (logits.argmax(-1) == labels).float().mean()
    return correspondence, gram, matches
