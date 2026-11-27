"""Native scalar RAW measurements with explicit sampling and sensor calibration.

An input may contain one monochrome photosite per pixel, a Bayer mosaic, or
co-located measurement planes. Filter identity and observed support are metadata;
unobserved spectral samples remain absent. No demosaicing or channel replication.
"""

import math

import torch
from torch import nn

from owac.raw.cfa import CFA_PATTERNS


def sensor_map(value, raw, sensor):
    """Broadcast scalar, per-plane, or global 2x2 Bayer-phase calibration."""
    value = torch.as_tensor(value, device=raw.device, dtype=torch.float32)
    if value.numel() == 1:
        return value
    if sensor["layout"] == "bayer":
        row, col = sensor.get("origin", (0, 0))
        y = (torch.arange(raw.shape[-2], device=raw.device) + row) % 2
        x = (torch.arange(raw.shape[-1], device=raw.device) + col) % 2
        return value.reshape(4)[y[:, None] * 2 + x[None, :]][None, None]
    return value.reshape(1, raw.shape[1], 1, 1)


def calibrated_measurement(raw, sensor):
    """Keep DN precision in FP32; quantization is measured in sensor units."""
    x = raw.float()
    black = sensor_map(sensor.get("black_level", 0), x, sensor)
    white = sensor_map(sensor["white_level"], x, sensor)
    span = white - black
    signal = (x - black) / span
    shot = sensor_map(sensor.get("shot", 0), x, sensor)
    read = sensor_map(sensor.get("read_var", 0), x, sensor)
    if sensor.get("noise_units", "normalized") == "dn":
        shot, read = shot / span, read / span.square()
    if sensor["units"] == "dn":
        quantum = 1 / span
    else:
        levels = sensor.get("levels", 0)
        quantum = (1 / levels) / span if levels else torch.zeros_like(span)
    variance = shot * signal.clamp_min(0) + read + quantum.square() / 12
    return signal, variance, x >= white


def attenuate_measurement(raw, stops, sensor, generator):
    """Approximate lower exposure in the sensor's units, including native DN.

    Existing read/shot noise is scaled first; only the missing variance is added.
    This is an augmentation of an already noisy/clipped sample, not independent
    Poisson thinning. DN rounding uses one digital count, never an 8-bit image.
    """
    signal, _, _ = calibrated_measurement(raw, sensor)
    black = sensor_map(sensor.get("black_level", 0), raw, sensor)
    white = sensor_map(sensor["white_level"], raw, sensor)
    span = white - black
    shot = sensor_map(sensor.get("shot", 0), raw, sensor)
    read = sensor_map(sensor.get("read_var", 0), raw, sensor)
    if sensor.get("noise_units", "normalized") == "dn":
        shot, read = shot / span, read / span.square()
    gain = 2.0 ** (-stops)
    added = gain * (1 - gain) * shot * signal.clamp_min(0) + (1 - gain**2) * read
    signal = gain * signal + added.sqrt() * torch.randn(
        signal.shape, generator=generator, device=signal.device, dtype=signal.dtype
    )
    result = torch.minimum((signal * span + black).clamp_min(0), white)
    if sensor["units"] == "dn":
        return result.round()
    levels = sensor.get("levels", 0)
    return (result * levels).round() / levels if levels else result


class MeasurementResponse(nn.Module):
    """Form sparse sensor-filter fields while retaining physical pixel positions.

    Packet fields, each with F planes: value, compressed value, nominal variance,
    observed support, and at/above-white indicator. F is a configurable vocabulary
    of sensor filter identities, independent of the number of input channels.
    """

    def __init__(self, cfg, pivot):
        super().__init__()
        self.sensor = cfg["sensor"]
        self.filter_names = tuple(cfg.get("filter_names", ("mono", "R", "G", "B")))
        self.shared_log_pivot = nn.Parameter(torch.tensor(math.log(pivot)))
        self.filter_log_pivot = nn.Parameter(torch.zeros(len(self.filter_names)))

    def forward(self, raw, *, sensor=None, valid=None):
        sensor = self.sensor if sensor is None else sensor
        if raw.ndim == 3:
            raw = raw[:, None]
        with torch.autocast(device_type=raw.device.type, enabled=False):
            signal, variance, saturated = calibrated_measurement(raw, sensor)
            if valid is not None and valid.ndim == 3:
                valid = valid[:, None]
            observed = (
                torch.ones_like(signal, dtype=torch.bool)
                if valid is None
                else torch.broadcast_to(valid.bool(), signal.shape)
            )
            signal = torch.where(observed, signal, 0)
            variance = torch.where(observed, variance, 0)
            if sensor["layout"] == "bayer":
                if raw.shape[1] != 1:
                    raise ValueError("Bayer RAW supplies one observed value per photosite")
                pattern = CFA_PATTERNS[sensor["pattern"].upper()]
                row, col = sensor.get("origin", (0, 0))
                y = (torch.arange(raw.shape[-2], device=raw.device) + row) % 2
                x = (torch.arange(raw.shape[-1], device=raw.device) + col) % 2
                phase = y[:, None] * 2 + x[None, :]
                ids = torch.tensor(
                    [self.filter_names.index(name) for name in pattern], device=raw.device
                )[phase][None, None]
            else:
                names = ["mono"] if sensor["layout"] == "mono" else sensor["filters"]
                if len(names) != raw.shape[1]:
                    raise ValueError("Every measured input plane needs its sensor-filter identity")
                ids = torch.tensor(
                    [self.filter_names.index(name) for name in names], device=raw.device
                )[None, :, None, None]
            fields = [[], [], [], [], []]
            pivots = (self.shared_log_pivot + self.filter_log_pivot).exp()
            for index, pivot in enumerate(pivots):
                support = observed & (ids == index)
                count = support.sum(1)
                # Counts describe actual observations, including repeated filter
                # measurements. Bayer's green phases occupy distinct positions.
                value = (signal * support).sum(1)
                noise = (variance * support).sum(1)
                compressed = (torch.asinh(signal / pivot) / torch.asinh(1 / pivot) * support).sum(1)
                fields[0].append(value)
                fields[1].append(compressed)
                fields[2].append(noise)
                fields[3].append(count.float())
                fields[4].append((saturated & support).sum(1).float())
            return torch.cat([torch.stack(field, 1) for field in fields], 1)


def unpack_measurements(packet, filters):
    return packet.split(filters, dim=1)
