# Copyright 2024 Big Vision Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
# See the License for the specific language governing permissions and limitations.
#
# Modified for OWAC: PyTorch OIHW kernels, nine-channel RAW response, differentiable
# application and explicit encoder-state conversion. See _vendor/flexivit_provenance.json.
"""FlexiViT PI-resize for owned RAW kernels; no implicit model/configuration change."""

from functools import cache

import torch
from torch.nn import functional as F

RAW_PATCH_WEIGHT = "backbone.embeddings.patch_embeddings.weight"


@cache
def _pi_operator(old_hw, new_hw):
    """Pseudoinverse of the transpose of half-pixel bilinear patch resizing."""
    pixels = old_hw[0] * old_hw[1]
    basis = torch.eye(pixels, dtype=torch.float64).reshape(pixels, 1, *old_hw)
    resized = F.interpolate(basis, size=new_hw, mode="bilinear", align_corners=False)
    resize_matrix = resized.flatten(1).T
    return torch.linalg.pinv(resize_matrix.T)


def pi_resize_patch_kernel(weight, new_hw):
    """Map an OIHW kernel to a new grid while retaining a gradient to the old kernel.

    A smaller patch gives a least-squares projection, not an invertible conversion.
    For repeated multiscale forwards, always start from the original parameter;
    never overwrite it with a succession of resized kernels. Bias is unchanged.
    """
    new_hw = tuple(new_hw)
    if weight.ndim != 4 or len(new_hw) != 2 or any(n <= 0 for n in new_hw):
        raise ValueError("Expected an OIHW kernel and two positive spatial dimensions.")
    old_hw = tuple(weight.shape[-2:])
    if old_hw == new_hw:
        return weight
    dtype = torch.float64 if weight.dtype == torch.float64 else torch.float32
    operator = _pi_operator(old_hw, new_hw).to(device=weight.device, dtype=dtype)
    result = F.linear(weight.to(dtype).flatten(2), operator)
    return result.reshape(*weight.shape[:2], *new_hw).to(weight.dtype)


def resized_raw_encoder_state(state, patch_size):
    """Return an encoder-only state for an explicitly configured new RawEncoder.

    This copies the mapping, changes only the patch kernel, and leaves every other
    tensor untouched. It does not migrate an optimizer or claim policy equivalence.
    """
    resized = dict(state)
    resized[RAW_PATCH_WEIGHT] = pi_resize_patch_kernel(
        state[RAW_PATCH_WEIGHT], (patch_size, patch_size)
    )
    return resized
