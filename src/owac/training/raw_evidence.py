"""Multiscale RAW measurement descriptors, not calibrated photon posteriors.

The variance law is the benchmark's signal-dependent Gaussian sensor model.
Pooling formulas assume independent pixels; interpolation, clipping and the
unknown clean signal make these nominal evidence descriptors after augmentation.
No RGB reconstruction, intensity threshold, or external visual weights are used.
"""

import torch
from torch import nn
from torch.nn import functional as F


class PhotonEvidencePyramid(nn.Module):
    """Retain spatial contrast, radiance, noise evidence and white-level proximity.

    Each original ViT patch contains fixed-location cells at several apertures.
    Their learned fusion depends on measured evidence, without discarding the
    original radiometric patch or imposing a brightness-based routing threshold.
    """

    def __init__(self, width, patch_size, cfg):
        super().__init__()
        self.patch_size = patch_size
        self.scales = tuple(cfg["scales"])
        if any(patch_size % scale for scale in self.scales):
            raise ValueError("Evidence cell sizes must divide the RAW patch size")
        self.shot = cfg["shot"]
        self.read_var = cfg["read_var"]
        self.quant_var = 1 / (12 * cfg["levels"] ** 2) if cfg["levels"] else 0.0
        self.white = cfg["white_level"]
        self.pivot = cfg["response_pivot"]
        self.projections = nn.ModuleList(
            nn.Sequential(nn.Linear(12 * (patch_size // scale) ** 2, width), nn.LayerNorm(width))
            for scale in self.scales
        )
        self.aperture = nn.Sequential(nn.Linear(6, 32), nn.SiLU(), nn.Linear(32, len(self.scales)))
        self.output = nn.Linear(width, width, bias=False)
        # Reuse the complete owned policy before learning a new measurement path.
        nn.init.zeros_(self.output.weight)

    def descriptors(self, raw):
        """Return FP32 descriptors and a per-patch aperture-conditioning vector."""
        with torch.autocast(device_type=raw.device.type, enabled=False):
            patch = self.patch_size
            height, width = raw.shape[-2:]
            # Match the existing non-overlapping Conv2d's incomplete-edge support.
            x = raw[..., : height // patch * patch, : width // patch * patch].float()
            variance = self.shot * x.clamp_min(0) + self.read_var + self.quant_var
            mean = F.avg_pool2d(x, patch, patch)
            mean_var = F.avg_pool2d(variance, patch, patch) / patch**2
            # Gaussian upper-tail proximity is a descriptor, not a posterior
            # probability that the latent (pre-clipped) radiance was saturated.
            white_proximity = 0.5 * torch.erfc((self.white - x) / (2 * variance).sqrt())
            proximity = F.avg_pool2d(white_proximity, patch, patch)
            evidence = torch.log1p(mean.square() / mean_var)
            aperture = torch.cat((evidence, proximity), 1).flatten(2).transpose(1, 2)
            descriptors = []
            for scale in self.scales:
                cells = patch // scale
                local = F.avg_pool2d(x, scale, scale)
                local_var = F.avg_pool2d(variance, scale, scale) / scale**2
                parent = mean.repeat_interleave(cells, -2).repeat_interleave(cells, -1)
                parent_var = mean_var.repeat_interleave(cells, -2).repeat_interleave(cells, -1)
                # Cov(child, parent) = (child_area / parent_area) Var(child).
                contrast_var = local_var * (1 - 2 / cells**2) + parent_var
                # The parent's uncertainty also regularizes an empty patch.
                contrast = (local - parent) / (parent.square() + contrast_var + parent_var).sqrt()
                radiance = torch.asinh(local / self.pivot)
                strength = torch.log1p(local.square() / local_var)
                white = F.avg_pool2d(white_proximity, scale, scale)
                features = torch.cat((contrast, radiance, strength, white), 1)
                # Unfold keeps every cell's fixed 2D position within its patch.
                descriptors.append(F.unfold(features, cells, stride=cells).transpose(1, 2))
        return descriptors, aperture

    def forward(self, raw):
        descriptors, evidence = self.descriptors(raw)
        alternatives = torch.stack(
            [
                projection(value)
                for projection, value in zip(self.projections, descriptors, strict=True)
            ],
            -2,
        )
        weights = self.aperture(evidence).softmax(-1)
        tokens = self.output((alternatives * weights[..., None]).sum(-2))
        height, width = raw.shape[-2:]
        return tokens.transpose(1, 2).reshape(
            len(raw), -1, height // self.patch_size, width // self.patch_size
        )


class EvidencePatchEmbedding(nn.Conv2d):
    """Keep original Conv2d checkpoint keys and add a RAW evidence computation."""

    def __init__(self, original, cfg):
        super().__init__(
            original.in_channels,
            original.out_channels,
            original.kernel_size,
            stride=original.stride,
            bias=original.bias is not None,
        )
        self.weight, self.bias = original.weight, original.bias
        self.evidence = PhotonEvidencePyramid(original.out_channels, original.kernel_size[0], cfg)

    def forward(self, response):
        return super().forward(response) + self.evidence(response[:, :3])
