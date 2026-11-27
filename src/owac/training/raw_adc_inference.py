"""Prototype local inference from quantized/censored scalar RAW observations.

The optional experimental patch encoder uses a fixed, nominal Gaussian
analog-noise approximation, not an exact photon posterior. Quantization enters
through integration over ADC bins, so its variance is not added a second time.
The caller supplies a spatial/filter observation matrix and a positive definite
quadratic prior. Coefficients are features, not a reconstructed RGB image.
"""

import math
from dataclasses import dataclass

import torch

from owac.raw.cfa import CFA_PATTERNS
from owac.training.raw_measurements import sensor_map


@dataclass
class ADCIntervals:
    signal: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    variance: torch.Tensor
    observed: torch.Tensor
    filter_ids: torch.Tensor
    point_mask: torch.Tensor


def native_adc_intervals(raw, sensor, *, valid=None, filter_names=("mono", "R", "G", "B")):
    """Preserve each native sample; missing samples have the entire real line.

    ``white_level`` is the sensor's clipping code, not a display white point.
    Native DN has unit-width bins. Normalized input requires explicit ``levels``;
    levels=0 describes a continuous measurement, not an invented ADC precision.
    Processed simulation input is not calibrated photon data in either case.
    Bayer phase calibration is evaluated at the actual input pixel coordinates.
    """
    if raw.ndim == 3:
        raw = raw[:, None]
    with torch.autocast(device_type=raw.device.type, enabled=False):
        value = raw.float()
        black = sensor_map(sensor.get("black_level", 0), value, sensor)
        white = sensor_map(sensor["white_level"], value, sensor)
        span = white - black
        levels = None if sensor["units"] == "dn" else sensor["levels"]
        quantum = 1.0 if sensor["units"] == "dn" else (1.0 / levels if levels else 0.0)
        signal = (value - black) / span
        lower = (value - quantum / 2 - black) / span
        upper = (value + quantum / 2 - black) / span
        lower = lower.masked_fill(value <= sensor.get("adc_min", 0), -torch.inf)
        upper = upper.masked_fill(value >= white, torch.inf)
        shot = sensor_map(sensor.get("shot", 0), value, sensor)
        read = sensor_map(sensor.get("read_var", 0), value, sensor)
        if sensor.get("noise_units", "normalized") == "dn":
            shot, read = shot / span, read / span.square()
        variance = shot * signal.clamp_min(0) + read
        if valid is not None and valid.ndim == 3:
            valid = valid[:, None]
        observed = (
            torch.ones_like(value, dtype=torch.bool)
            if valid is None
            else torch.broadcast_to(valid.bool(), value.shape)
        )
        if torch.any(observed & (variance <= 0)):
            raise ValueError("The Gaussian interval prototype requires positive analog variance")
        lower = lower.masked_fill(~observed, -torch.inf)
        upper = upper.masked_fill(~observed, torch.inf)
        variance = torch.where(observed, variance, 1)
        point_mask = observed & torch.isfinite(lower) & torch.isfinite(upper) & (quantum == 0)
        if sensor["layout"] == "bayer":
            if value.shape[1] != 1:
                raise ValueError("Bayer input has one measurement per photosite")
            row, col = sensor.get("origin", (0, 0))
            y = (torch.arange(value.shape[-2], device=value.device) + row) % 2
            x = (torch.arange(value.shape[-1], device=value.device) + col) % 2
            ids = torch.tensor(
                [filter_names.index(name) for name in CFA_PATTERNS[sensor["pattern"].upper()]],
                device=value.device,
            )[y[:, None] * 2 + x[None, :]][None, None]
        else:
            names = ["mono"] if sensor["layout"] == "mono" else sensor["filters"]
            if len(names) != value.shape[1]:
                raise ValueError("Each measured input plane needs its filter identity")
            ids = torch.tensor([filter_names.index(name) for name in names], device=value.device)[
                None, :, None, None
            ]
        return ADCIntervals(
            signal, lower, upper, variance, observed, ids.expand_as(value), point_mask
        )


def gaussian_interval_statistics(lower, upper, mean, variance):
    """Log probability, conditional analog mean, and observed mean precision.

    Fixed positive variance is assumed. Tail moment ratios are computed in FP64:
    FP32 log-density subtraction can put a 10bit saturated conditional mean
    outside its observation interval. Outputs retain the caller's dtype.
    CDF differences use the nearer tail and log/expm1 arithmetic. Precision
    is the negative second derivative of log bin probability with respect to
    mean, not arbitrary feature variance. Its Gaussian interval bounds [0, 1/v]
    are applied only to remove floating-point cancellation outside those bounds.
    """
    dtype = mean.dtype
    values = (lower.double(), upper.double(), mean.double(), variance.double())
    return tuple(value.to(dtype) for value in _gaussian_interval_statistics(*values))


def _gaussian_interval_statistics(lower, upper, mean, variance):
    sigma = variance.sqrt()
    finite_lo, finite_hi = torch.isfinite(lower), torch.isfinite(upper)
    safe_lo = ((lower.masked_fill(~finite_lo, 0) - mean) / sigma).masked_fill(~finite_lo, 0)
    safe_hi = ((upper.masked_fill(~finite_hi, 0) - mean) / sigma).masked_fill(~finite_hi, 0)
    # On the positive half-line use survival probabilities to avoid 1 - 1.
    right = safe_lo > 0

    def log_cdf_at_bound(safe, bound):
        result = torch.special.log_ndtr(safe)
        result = result.masked_fill(torch.isneginf(bound), -torch.inf)
        return result.masked_fill(torch.isposinf(bound), 0)

    big = torch.where(right, log_cdf_at_bound(-safe_lo, -lower), log_cdf_at_bound(safe_hi, upper))
    small = torch.where(right, log_cdf_at_bound(-safe_hi, -upper), log_cdf_at_bound(safe_lo, lower))
    log_mass = big + torch.log(-torch.expm1(small - big))
    log_normalizer = 0.5 * math.log(2 * math.pi)
    log_lo_density = (-safe_lo.square() / 2 - log_normalizer).masked_fill(~finite_lo, -torch.inf)
    log_hi_density = (-safe_hi.square() / 2 - log_normalizer).masked_fill(~finite_hi, -torch.inf)
    lo_ratio, hi_ratio = (log_lo_density - log_mass).exp(), (log_hi_density - log_mass).exp()
    shift = lo_ratio - hi_ratio
    conditional_mean = mean + sigma * shift
    normalized_precision = shift.square() - (safe_lo * lo_ratio - safe_hi * hi_ratio)
    precision = normalized_precision.clamp(0, 1) / variance
    return log_mass, conditional_mean, precision


def gaussian_measurement_statistics(lower, upper, mean, variance, point_values, point_mask):
    """Mixed interval probability and explicit continuous-observation density.

    A continuous interior observation is a point density, never a zero-width
    interval probability. Censored endpoints still use one-sided integration.
    Values are only comparable under a fixed observation model and units.
    """
    if point_mask is None:
        return gaussian_interval_statistics(lower, upper, mean, variance)
    lower = lower.masked_fill(point_mask, -torch.inf)
    upper = upper.masked_fill(point_mask, torch.inf)
    mass, analog_mean, precision = gaussian_interval_statistics(lower, upper, mean, variance)
    point_values = torch.where(point_mask, point_values, mean)
    density = -0.5 * ((point_values - mean).square() / variance + (2 * math.pi * variance).log())
    return (
        torch.where(point_mask, density, mass),
        torch.where(point_mask, point_values, analog_mean),
        torch.where(point_mask, 1 / variance, precision),
    )


def infer_interval_coefficients(
    lower,
    upper,
    variance,
    observed,
    design,
    prior_mean,
    prior_precision,
    *,
    iterations=4,
    point_values=None,
    point_mask=None,
):
    """Fixed-count EM steps for a linear mean and a quadratic coefficient prior.

    Shapes: observations [..., samples], design [..., samples, coefficients],
    prior mean [..., coefficients], prior precision [..., coefficients, coefficients].
    Inputs may broadcast. Each call starts from its supplied prior; iterations
    do not accumulate independent evidence. No convergence or calibrated
    posterior claim is made for finite steps or the nominal sensor-noise model.
    """
    dtype = torch.float64 if design.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=design.device.type, enabled=False):
        return _infer_interval_coefficients(
            lower.to(dtype),
            upper.to(dtype),
            variance.to(dtype),
            observed,
            design.to(dtype),
            prior_mean.to(dtype),
            prior_precision.to(dtype),
            iterations,
            point_values.to(dtype) if point_values is not None else None,
            point_mask & observed if point_mask is not None else None,
        )


def _infer_interval_coefficients(
    lower,
    upper,
    variance,
    observed,
    design,
    prior_mean,
    prior_precision,
    iterations,
    point_values,
    point_mask,
):
    lower = torch.where(observed, lower, -torch.inf)
    upper = torch.where(observed, upper, torch.inf)
    variance = torch.where(observed, variance, 1)
    weights = observed.to(variance.dtype) / variance
    transpose = design.transpose(-1, -2)
    information = transpose @ (weights[..., None] * design)
    factor = torch.linalg.cholesky(prior_precision + information)
    prior_rhs = prior_precision @ prior_mean[..., None]
    coefficients = prior_mean
    for _ in range(iterations):
        mean = (design @ coefficients[..., None]).squeeze(-1)
        _, analog_mean, _ = gaussian_measurement_statistics(
            lower, upper, mean, variance, point_values, point_mask
        )
        rhs = prior_rhs + transpose @ (weights * analog_mean)[..., None]
        coefficients = torch.cholesky_solve(rhs, factor).squeeze(-1)
    mean = (design @ coefficients[..., None]).squeeze(-1)
    log_mass, _, precision = gaussian_measurement_statistics(
        lower, upper, mean, variance, point_values, point_mask
    )
    data_precision = transpose @ ((precision * observed)[..., None] * design)
    return {
        "coefficients": coefficients,
        "measurement_precision": data_precision,
        "prior_precision": prior_precision,
        "log_likelihood": (log_mass * observed).sum(-1),
        "observed_samples": observed.sum(-1),
    }
