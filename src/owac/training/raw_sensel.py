"""Patch features from scalar sensor measurements and their observed support.

All computation is independent of input channel count. Sparse Bayer samples,
monochrome photosites and dense spectral planes use the same operators. The
measurement front end stays in FP32 before entering the shared Transformer.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from owac.training.raw_measurements import unpack_measurements


def cell_sum(value, size):
    return F.avg_pool2d(value, size, size) * size**2


class SenselEvidencePyramid(nn.Module):
    """Shared per-filter statistics, aperture fusion and observed-filter pooling."""

    def __init__(self, width, patch, filters, scales, pivot):
        super().__init__()
        self.patch, self.filters, self.scales, self.pivot = patch, filters, scales, pivot
        self.projections = nn.ModuleList(
            nn.Sequential(nn.Linear(8 * (patch // scale) ** 2, width), nn.LayerNorm(width))
            for scale in scales
        )
        self.filter_condition = nn.Embedding(filters, 8)
        self.filter_embedding = nn.Embedding(filters, width)
        self.aperture = nn.Sequential(nn.Linear(12, 32), nn.SiLU(), nn.Linear(32, len(scales)))
        self.filter_score = nn.Linear(width, 1)
        self.output = nn.Linear(width, width, bias=False)
        nn.init.zeros_(self.output.weight)

    def descriptors(self, packet):
        value, _, variance, observed, saturated = unpack_measurements(packet, self.filters)
        patch = self.patch
        height, width = value.shape[-2:]
        count = cell_sum(observed, patch)
        denominator = count.clamp_min(1)
        mean = cell_sum(value, patch) / denominator
        mean_var = cell_sum(variance, patch) / denominator.square()
        tiny = torch.finfo(torch.float32).tiny
        # Distance to the sum of the white levels of the observations at a site.
        proximity = 0.5 * torch.erfc((observed - value) / (2 * variance).clamp_min(tiny).sqrt())
        proximity = proximity * (observed > 0)
        headroom = cell_sum(proximity, patch) / denominator
        clipped = cell_sum(saturated, patch) / denominator
        strength = torch.log1p(mean.square() / mean_var.clamp_min(tiny))
        context = torch.stack((strength, headroom, clipped, count / patch**2), -1)
        context = context.flatten(2, 3).permute(0, 2, 1, 3)
        x = (torch.arange(width, device=value.device).float() % patch + 0.5) / patch
        y = (torch.arange(height, device=value.device).float() % patch + 0.5) / patch
        positions = (observed * x[None, None, None, :], observed * y[None, None, :, None])
        result = []
        for scale in self.scales:
            cells = patch // scale
            local_count = cell_sum(observed, scale)
            local_denominator = local_count.clamp_min(1)
            local_mean = cell_sum(value, scale) / local_denominator
            local_var = cell_sum(variance, scale) / local_denominator.square()

            def expand(z, repeats=cells):
                return z.repeat_interleave(repeats, -2).repeat_interleave(repeats, -1)

            parent, parent_var, parent_count = expand(mean), expand(mean_var), expand(denominator)
            contrast_var = (
                local_var + parent_var - 2 * local_var * local_count / parent_count
            ).clamp_min(0)
            contrast = (local_mean - parent) / (
                parent.square() + contrast_var + parent_var
            ).clamp_min(tiny).sqrt()
            fields = [
                contrast,
                torch.asinh(local_mean / self.pivot),
                torch.log1p(local_mean.square() / local_var.clamp_min(tiny)),
                cell_sum(proximity, scale) / local_denominator,
                cell_sum(saturated, scale) / local_denominator,
                local_count / scale**2,
                *(cell_sum(position, scale) / local_denominator for position in positions),
            ]
            features = torch.stack(fields, 2) * (local_count > 0)[:, :, None]
            batch, filters = features.shape[:2]
            tokens = F.unfold(features.flatten(0, 1), cells, stride=cells).transpose(1, 2)
            # Dense [batch, patch, filter, feature] rows let Linear flatten its
            # leading dimensions into one GEMM. A transposed view dispatches
            # frozen inference weights to strided batched GEMM, which fails on
            # the CUDA 13.2 runtime used by the native training validation.
            result.append(
                tokens.reshape(batch, filters, tokens.shape[1], -1).transpose(1, 2).contiguous()
            )
        valid = count.flatten(2).transpose(1, 2) > 0
        return result, context, valid

    def forward(self, packet):
        descriptors, context, valid = self.descriptors(packet)
        condition = self.filter_condition.weight[None, None].expand(*context.shape[:-1], -1)
        gates = self.aperture(torch.cat((context, condition), -1)).softmax(-1)
        choices = torch.stack(
            [layer(x) for layer, x in zip(self.projections, descriptors, strict=True)], -2
        )
        per_filter = (choices * gates[..., None]).sum(-2) + self.filter_embedding.weight
        score = self.filter_score(per_filter).squeeze(-1).masked_fill(~valid, -torch.inf)
        score = torch.where(valid.any(-1, keepdim=True), score, 0)
        weights = score.softmax(-1) * valid
        tokens = self.output((per_filter * weights[..., None]).sum(-2))
        height, width = packet.shape[-2:]
        return tokens.transpose(1, 2).reshape(
            len(packet), -1, height // self.patch, width // self.patch
        )


class SenselPatchEmbedding(nn.Module):
    """Shared scalar-response kernels plus sensor-filter adaptations.

    A kernel is applied only at actually observed photosites. Per-filter sample
    count normalizes its contribution; the separate evidence path retains the
    amount, noise and geometry of the supporting measurements.
    """

    def __init__(self, width, patch, cfg):
        super().__init__()
        self.patch = patch
        self.filter_names = tuple(cfg.get("filter_names", ("mono", "R", "G", "B")))
        self.shared_weight = nn.Parameter(torch.empty(width, 2, patch, patch))
        self.filter_weight = nn.Parameter(
            torch.zeros(len(self.filter_names), width, 2, patch, patch)
        )
        self.bias = nn.Parameter(torch.zeros(width))
        nn.init.kaiming_uniform_(self.shared_weight, a=math.sqrt(5))
        self.evidence = (
            SenselEvidencePyramid(
                width, patch, len(self.filter_names), cfg["scales"], cfg["response_pivot"]
            )
            if cfg.get("evidence", True)
            else None
        )

    @property
    def weight(self):
        # HuggingFace uses this property only to select the input dtype.
        return self.shared_weight

    def forward(self, packet):
        with torch.autocast(device_type=packet.device.type, enabled=False):
            packet = packet.float()
            patch, filters = self.patch, len(self.filter_names)
            height, width = packet.shape[-2:]
            packet = packet[..., : height // patch * patch, : width // patch * patch]
            value, compressed, _, observed, _ = unpack_measurements(packet, filters)
            features = torch.stack((value, compressed), 2).flatten(1, 2)
            kernels = (self.shared_weight[None] + self.filter_weight).flatten(0, 1)
            projected = F.conv2d(features, kernels, stride=patch, groups=filters)
            projected = projected.reshape(len(packet), filters, -1, *projected.shape[-2:])
            count = cell_sum(observed, patch)
            scale = patch**2 / count.clamp_min(1) * (count > 0)
            output = (projected * scale[:, :, None]).sum(1) + self.bias[None, :, None, None]
            if self.evidence is not None:
                output = output + self.evidence(packet)
            return output


def migrate_owned_rgb_projection(encoder, weights, prefix=""):
    """Compile the OLD owned three-plane response into generic filter kernels.

    This function is an initialization adapter, not the new encoder computation.
    The old relative-channel response is eliminated algebraically. Monochrome
    starts from the shared mean kernel and can subsequently be adapted directly.
    """
    module = encoder.backbone.embeddings.patch_embeddings
    key = prefix + "backbone.embeddings.patch_embeddings."
    if not isinstance(module, SenselPatchEmbedding) or key + "weight" not in weights:
        return None
    original = weights.pop(key + "weight")
    width, _, height, wid = original.shape
    parts = original.reshape(width, 3, 3, height, wid)
    compressed = parts[:, 1] + parts[:, 2] - parts[:, 2].mean(1, keepdim=True)
    kernels = torch.stack((parts[:, 0], compressed), 2).transpose(0, 1)
    shared = kernels.mean(0)
    residual = torch.zeros_like(module.filter_weight, device=original.device)
    pivot = weights.pop(prefix + "response.log_pivot").reshape(3)
    mean_pivot = pivot.mean()
    pivot_residual = torch.zeros_like(encoder.response.filter_log_pivot, device=pivot.device)
    for index, name in enumerate(("R", "G", "B")):
        if name not in module.filter_names:
            continue
        target = module.filter_names.index(name)
        residual[target] = kernels[index] - shared
        pivot_residual[target] = pivot[index] - mean_pivot
    weights[key + "shared_weight"] = shared
    weights[key + "filter_weight"] = residual
    weights[prefix + "response.shared_log_pivot"] = mean_pivot
    weights[prefix + "response.filter_log_pivot"] = pivot_residual
    return {
        "source": "owned three-plane radiometric projection",
        "filters": module.filter_names,
        "monochrome_initialization": "shared mean scalar-response kernel; no channel replication",
        "precision": "FP32 measurement front end; algebraic transfer, not bitwise BF16 equivalence",
    }
