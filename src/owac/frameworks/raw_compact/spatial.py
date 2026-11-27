"""Two-dimensional sinusoidal readout coordinates, following ACT/DETR practice."""

import math

import torch
from torch.nn import functional as F

from owac.frameworks.raw_compact.task_mlp import expand_shared_mlp_weights


def resize_raw_views(raw, size=None, *, sensor=None):
    """Resample RAW consistently inside the policy; no radiometric clipping."""
    if size is None or tuple(size) == raw.shape[-2:]:
        return raw
    if sensor and (sensor["layout"] in ("mono", "bayer") or sensor["units"] == "dn"):
        raise ValueError(
            "Native RAW uses its physical sampling grid; choose the patch/readout grid "
            "instead of interpolating sensor measurements with raw_image_size"
        )
    batch, views, channels = raw.shape[:3]
    resized = F.interpolate(
        raw.flatten(0, 1), size=size, mode="bilinear", align_corners=False, antialias=True
    )
    return resized.reshape(batch, views, channels, *size)


def position_encoding(height, width, channels, *, device, dtype):
    # Four equal sin/cos groups encode row and column coordinates.
    if channels % 4:
        raise ValueError("Two-dimensional sinusoidal encoding needs a multiple of 4 channels")
    frequency = 10000 ** (-torch.arange(channels // 4, device=device).float() / (channels // 4))
    row = (torch.arange(height, device=device).float() + 0.5) / height * (2 * math.pi)
    col = (torch.arange(width, device=device).float() + 0.5) / width * (2 * math.pi)
    yy, xx = torch.meshgrid(row, col, indexing="ij")
    y, x = yy.flatten()[:, None] * frequency, xx.flatten()[:, None] * frequency
    return torch.cat((y.sin(), y.cos(), x.sin(), x.cos()), -1).to(dtype)


def inherit_policy(model, payload, *, patch_resize=None):
    """Warm-start a new experiment from a selected policy, not an optimizer resume.

    The C2 expansion adds state coordinates and a zero-initialized spatial readout.
    An explicit PI resize changes only the RAW patch kernel for a new fixed grid.
    New policy parameters retain their constructor initialization, including
    optional zero-gated observation refinement. Task-specific FFNs copy each
    corresponding shared FFN onto the task axis. Other policy weights are preserved;
    incompatible tensors still fail through the ordinary load_state_dict shape
    contract. Optimizer state is not inherited.
    """
    weights = dict(payload["ema"] if "ema" in payload else payload["model"])
    report = {"source_step": payload["step"]}
    from owac.training.raw_sensel import migrate_owned_rgb_projection

    measurement = (
        migrate_owned_rgb_projection(model.raw_encoder, weights, "raw_encoder.")
        if "raw_encoder.backbone.embeddings.patch_embeddings.weight" in weights
        else None
    )
    if measurement is not None:
        report["measurement_projection"] = measurement
    expanded_mlps = expand_shared_mlp_weights(model.action_generator, weights)
    if expanded_mlps:
        report["task_mlp_expansion"] = expanded_mlps
    if patch_resize is not None:
        from owac.training.raw_patch_resize import RAW_PATCH_WEIGHT, pi_resize_patch_kernel

        if patch_resize != "pi":
            raise ValueError(f"Unknown policy patch resize: {patch_resize}")
        key = "raw_encoder." + RAW_PATCH_WEIGHT
        old_hw = tuple(weights[key].shape[-2:])
        new_hw = tuple(model.raw_encoder.backbone.embeddings.patch_embeddings.weight.shape[-2:])
        weights[key] = pi_resize_patch_kernel(weights[key], new_hw)
        report["patch_resize"] = {"method": "pi", "source_hw": old_hw, "target_hw": new_hw}
    key = "state_encoder.projection.weight"
    previous = weights[key]
    current = model.state_encoder.projection.weight
    if current.shape[1] > previous.shape[1]:
        expanded = previous.new_zeros(current.shape)
        expanded[:, : previous.shape[1]] = previous
        weights[key] = expanded
    initial = model.state_dict()
    new = sorted(set(initial) - set(weights))
    initial.update(weights)
    model.load_state_dict(initial)
    return {**report, "new_state_tensors": new}
