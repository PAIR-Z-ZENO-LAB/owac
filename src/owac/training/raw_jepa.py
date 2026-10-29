"""RAW-only masked feature prediction, initialized from public pi05 vision weights.

No action model, RGB target, external RAW teacher or historical student is loaded.
The target encoder is the EMA of this run's online encoder.
"""

from __future__ import annotations

import copy
import math
import random

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def block_mask(grid: int, rng: random.Random, blocks=4, scale=(0.15, 0.20)):
    """Shared spatial mask across every frame; context excludes target union."""
    while True:
        mask = torch.zeros(grid, grid, dtype=torch.bool)
        for _ in range(blocks):
            area = rng.uniform(*scale) * grid * grid
            aspect = rng.uniform(0.75, 1.5)
            height = min(grid, max(1, round(math.sqrt(area / aspect))))
            width = min(grid, max(1, round(math.sqrt(area * aspect))))
            top, left = rng.randrange(grid - height + 1), rng.randrange(grid - width + 1)
            mask[top : top + height, left : left + width] = True
        if not mask.all():
            return mask.flatten()


def hide_pixels(raw, mask, patch_size):
    """Mask before the spatial stem, not after its overlapping 3x3 convolution."""
    grid = math.isqrt(mask.numel())
    hidden = mask.reshape(grid, grid).repeat_interleave(patch_size, 0)
    hidden = hidden.repeat_interleave(patch_size, 1)
    return raw.masked_fill(hidden[None, None, None], 0)


def sincos_positions(grid, width):
    y, x = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    frequencies = 1 / (10000 ** (torch.arange(width // 4).float() / (width // 4)))
    return torch.cat(
        [(p.flatten()[:, None] * frequencies).sin() for p in (y, x)]
        + [(p.flatten()[:, None] * frequencies).cos() for p in (y, x)],
        dim=-1,
    )


class RawVisualEncoder(nn.Module):
    """Six-frame causal gated stem, then inherited SigLIP (pre-projector features).

    State is local to each window and reset on every forward: no previous source,
    unmasked branch, or future observation can enter the online context.
    """

    def __init__(self, tower, *, checkpoint_blocks=True, refinement_layer=None):
        super().__init__()
        self.vision = tower.vision_model
        self.width = self.vision.config.hidden_size
        self.patch_size = self.vision.config.patch_size
        self.gate_logit = nn.Parameter(torch.zeros(self.width))
        self.checkpoint_blocks = checkpoint_blocks
        self.refinement_layer = refinement_layer
        if refinement_layer is not None:
            self.refinement_scale = nn.Parameter(torch.zeros(()))
        self.requires_grad_(True)

    def forward(self, raw, mask=None, *, stop_at=None):
        # raw [batch*camera, six frames, 3, 224, 224]; no persistent state.
        if mask is not None:
            raw = hide_pixels(raw, mask, self.patch_size)
        batch, frames = raw.shape[:2]
        tokens = self.vision.embeddings.patch_embedding(raw.flatten(0, 1))
        tokens = tokens.flatten(2).transpose(1, 2).reshape(batch, frames, -1, self.width)
        gate = self.gate_logit.sigmoid().to(tokens.dtype)
        hidden = tokens[:, 0]
        for index in range(1, frames):
            hidden = gate * tokens[:, index] + (1 - gate) * hidden
        hidden = hidden + self.vision.embeddings.position_embedding.weight.to(hidden.dtype)
        if mask is not None:
            hidden = hidden[:, ~mask]
        for depth, layer in enumerate(self.vision.encoder.layers, 1):

            def forward(value, layer=layer):
                return layer(value, attention_mask=None)[0]

            if self.training and self.checkpoint_blocks and torch.is_grad_enabled():
                hidden = checkpoint(forward, hidden, use_reentrant=False)
            else:
                hidden = forward(hidden)
            if depth == stop_at:
                return hidden
            if depth == self.refinement_layer:
                spatial = hidden
        if self.refinement_layer is not None:
            # Start at the measured useful layer; later blocks learn a residual.
            # No final LayerNorm on the skip: step zero preserves its cosine geometry.
            return spatial + self.refinement_scale.tanh() * self.vision.post_layernorm(hidden)
        return self.vision.post_layernorm(hidden)


class SpatialPredictor(nn.Module):
    def __init__(self, input_width=1152, width=384, depth=4, heads=6, grid=16):
        super().__init__()
        self.input = nn.Linear(input_width, width)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.normal_(self.mask_token, std=0.02)
        self.register_buffer("positions", sincos_positions(grid, width), persistent=True)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    width,
                    heads,
                    width * 4,
                    dropout=0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, input_width)

    def forward(self, context, mask):
        projected = self.input(context)
        values = self.mask_token.to(projected.dtype).expand(context.shape[0], len(mask), -1).clone()
        values[:, ~mask] = projected
        values = values + self.positions.to(values.dtype)
        for layer in self.blocks:
            values = layer(values)
        return self.output(self.norm(values[:, mask]))


class RawJEPA(nn.Module):
    def __init__(self, encoder, predictor):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor

    def forward(self, raw, mask, views=None):
        prediction = self.predictor(self.encoder(raw, mask), mask)
        if views is not None:
            return prediction, self.encoder(views[0]), self.encoder(views[1])
        return prediction


def build_jepa(config, device):
    from owac.baselines.raw_stem_siglip import LocalConvStem
    from owac.raw.all_bench_distillation import _native_vision

    tower, projector = _native_vision(config, device="cpu")
    cfg = config["student"]
    arch = cfg["vision_architecture"]
    tower.vision_model.embeddings.patch_embedding = LocalConvStem(
        3,
        arch["hidden_size"],
        arch["patch_size"],
        conv_channels=cfg["stem_hidden"],
        seed=cfg["stem_seed"],
        dtype=torch.float32,
        device="cpu",
    )
    encoder = RawVisualEncoder(tower, refinement_layer=cfg.get("refinement_layer"))
    if config.get("initial_encoder"):
        state = torch.load(config["initial_encoder"], map_location="cpu", weights_only=False)
        weights = dict(state["encoder"])
        # A new experiment, not an exact optimizer resume of the old objective.
        if cfg.get("refinement_layer") is not None and "refinement_scale" not in weights:
            weights["refinement_scale"] = torch.zeros(())
        encoder.load_state_dict(weights, strict=True)
        del state, weights
    target = copy.deepcopy(encoder).requires_grad_(False).eval().to(device)
    predictor = SpatialPredictor(input_width=arch["hidden_size"], **config["predictor"])
    online = RawJEPA(encoder, predictor).to(device)
    # Retained unchanged for subsequent policy integration; never used in the loss.
    return online, target, projector


@torch.no_grad()
def update_target(target, encoder, momentum):
    for ema, source in zip(target.parameters(), encoder.parameters(), strict=True):
        ema.lerp_(source, 1 - momentum)


def prediction_loss(prediction, features, mask):
    target = F.layer_norm(features.float(), (features.shape[-1],))[:, mask].detach()
    return F.smooth_l1_loss(prediction.float(), target), target


@torch.no_grad()
def representation_metrics(prediction, target, full_features):
    features = F.layer_norm(full_features.float(), (full_features.shape[-1],))
    pooled = features.mean(1)
    spatial_mean = features.mean(1, keepdim=True).expand_as(features)
    return {
        "prediction_cosine": F.cosine_similarity(prediction.float(), target, dim=-1).mean(),
        "target_spatial_std": features.std(dim=1, correction=0).mean(),
        "target_between_image_std": pooled.std(dim=0, correction=0).mean(),
        "target_spatial_cosine": F.cosine_similarity(features, features.roll(1, 1), dim=-1).mean(),
        "target_between_image_cosine": F.cosine_similarity(
            pooled, pooled.roll(1, 0), dim=-1
        ).mean(),
        "spatial_mean_baseline": F.smooth_l1_loss(spatial_mean, features),
    }
