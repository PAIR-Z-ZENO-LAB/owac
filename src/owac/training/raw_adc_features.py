"""Learn spatial RAW coefficients from individual scalar ADC measurements.

The optional residual preserves the owned scalar projection at initialization.
It does not demosaic or reconstruct RGB. A point-observation control has exactly
the same learned parameters; only its observation likelihood differs.
"""

import torch
from torch import nn
from torch.nn import functional as F

from owac.training.raw_adc_inference import infer_interval_coefficients, native_adc_intervals
from owac.training.raw_measurements import MeasurementResponse
from owac.training.raw_sensel import SenselPatchEmbedding


class ADCMeasurementResponse(MeasurementResponse):
    """Append unsummed observations to the legacy shared-filter response packet.

    The first 5F fields remain unchanged. Seven subsequent groups have C planes each:
    signal, lower/upper bound, variance, support, filter ID and continuous-point mask.
    C is the actual
    number of measurements at a photosite, including repeated filters. Missing data
    is never substituted with a measured black sample. This is an internal tensor
    packet, not a multi-channel representation required of the native sensor.
    """

    def forward(self, raw, *, sensor=None, valid=None):
        sensor = self.sensor if sensor is None else sensor
        base = super().forward(raw, sensor=sensor, valid=valid)
        adc = native_adc_intervals(raw, sensor, valid=valid, filter_names=self.filter_names)
        fields = (
            adc.signal.masked_fill(~adc.observed, 0),
            adc.lower,
            adc.upper,
            adc.variance,
            adc.observed.float(),
            adc.filter_ids.float(),
            adc.point_mask.float(),
        )
        return torch.cat((base, *fields), 1)


class ADCSpatialFeatures(nn.Module):
    """Task-learnable local basis, observation-conditioned regularizer and readout.

    Each filter has its own coefficients on a shared spatial basis. The basis is
    evaluated only at actual measured positions. Its initialization is the six low
    2D polynomial modes; the pixel values of the modes subsequently learn directly.
    The regularizer depends on the same patch, so it is not an independent Bayesian
    prior. Returned measurement curvature excludes the regularizer precision.
    """

    def __init__(self, width, patch, filters, cfg):
        super().__init__()
        self.patch, self.filters = patch, filters
        self.likelihood = cfg["likelihood"]
        if self.likelihood not in ("interval", "point"):
            raise ValueError(f"Unknown ADC likelihood: {self.likelihood}")
        self.iterations = cfg.get("iterations", 4)
        coordinate = (torch.arange(patch).float() + 0.5) / patch * 2 - 1
        y, x = torch.meshgrid(coordinate, coordinate, indexing="ij")
        modes = torch.stack((torch.ones_like(x), x, y, x * y, x.square(), y.square()), -1)
        modes[..., 4:] -= modes[..., 4:].mean((0, 1))
        modes /= modes.square().mean((0, 1)).sqrt()
        self.basis = nn.Parameter(modes.flatten(0, 1))
        self.coefficients = modes.shape[-1]
        self.prior = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, filters * self.coefficients)
        )
        nn.init.zeros_(self.prior[-1].weight)
        nn.init.zeros_(self.prior[-1].bias)
        self.log_precision = nn.Parameter(torch.zeros(filters * self.coefficients))
        hidden = cfg.get("width", 128)
        self.features = nn.Sequential(
            nn.Linear(2 * self.coefficients + 1, hidden), nn.SiLU(), nn.LayerNorm(hidden)
        )
        self.filter_embedding = nn.Embedding(filters, hidden)
        self.output = nn.Linear(hidden, width, bias=False)
        nn.init.zeros_(self.output.weight)

    def infer(self, packet, base):
        """Expose coefficients/support for mechanism analysis, not a success metric."""
        patch, filters, coefficients = self.patch, self.filters, self.coefficients
        batch, _, height, width = packet.shape
        channels = packet.shape[1] // 7
        # Pixel ordering matches basis repetition for every original plane.
        samples = F.unfold(packet, patch, stride=patch).transpose(1, 2)
        samples = samples.reshape(batch, -1, 7, channels * patch**2)
        signal, lower, upper, variance, support, ids, points = samples.unbind(2)
        observed = support.bool()
        identity = F.one_hot(ids.long(), filters).to(signal.dtype)
        basis = self.basis.repeat(channels, 1)
        design = (identity[..., None] * basis[:, None, :]).flatten(-2)
        count = (identity * support[..., None]).sum(-2)
        means = (identity * signal[..., None]).sum(-2) / count.clamp_min(1)
        center = torch.zeros(*means.shape, coefficients, device=base.device, dtype=base.dtype)
        center[..., 0] = means
        prior = center.flatten(-2) + self.prior(base.flatten(2).transpose(1, 2).contiguous())
        precision = self.log_precision.exp().diag()
        if self.likelihood == "interval":
            result = infer_interval_coefficients(
                lower,
                upper,
                variance,
                observed,
                design,
                prior,
                precision,
                iterations=self.iterations,
                point_values=signal,
                point_mask=points.bool(),
            )
        else:
            # Ordinary weighted regression regards the clipped code as a point.
            # Finite ADC bins add q^2/12; clipped bins recover the same quantum
            # from adjacent spacing encoded by their finite endpoint and signal.
            quantum = torch.where(
                torch.isfinite(lower) & torch.isfinite(upper),
                upper - lower,
                torch.where(torch.isfinite(lower), 2 * (signal - lower), 2 * (upper - signal)),
            )
            quantum = torch.where(observed, quantum, 0)
            weights = support / (variance + quantum.square() / 12)
            transpose = design.transpose(-1, -2)
            information = transpose @ (weights[..., None] * design)
            rhs = precision @ prior[..., None] + transpose @ (weights * signal)[..., None]
            factor = torch.linalg.cholesky(precision + information)
            result = {
                "coefficients": torch.cholesky_solve(rhs, factor).squeeze(-1),
                "measurement_precision": information,
                "prior_precision": precision,
                "observed_samples": support.sum(-1),
            }
        result["filter_samples"] = count
        result["grid"] = (height // patch, width // patch)
        return result

    def forward(self, packet, base):
        inferred = self.infer(packet, base)
        count = inferred["filter_samples"]
        shape = (*count.shape, self.coefficients)
        coefficients = inferred["coefficients"].reshape(shape)
        information = inferred["measurement_precision"].diagonal(dim1=-2, dim2=-1).reshape(shape)
        fields = torch.cat(
            (torch.asinh(coefficients), torch.log1p(information), torch.log1p(count)[..., None]),
            -1,
        ).contiguous()
        features = self.features(fields) + self.filter_embedding.weight
        present = count > 0
        pooled = (features * present[..., None]).sum(-2) / present.sum(-1, keepdim=True).clamp_min(
            1
        )
        output = self.output(pooled)
        return output.transpose(1, 2).reshape(len(packet), -1, *inferred["grid"])


class ADCPatchEmbedding(SenselPatchEmbedding):
    def __init__(self, width, patch, cfg):
        super().__init__(width, patch, cfg)
        self.adc = ADCSpatialFeatures(width, patch, len(self.filter_names), cfg["adc_features"])

    def forward(self, packet):
        with torch.autocast(device_type=packet.device.type, enabled=False):
            height, width = packet.shape[-2:]
            packet = packet.float()[
                ..., : height // self.patch * self.patch, : width // self.patch * self.patch
            ]
            base_channels = 5 * len(self.filter_names)
            base = super().forward(packet[:, :base_channels])
            return base + self.adc(packet[:, base_channels:], base)
