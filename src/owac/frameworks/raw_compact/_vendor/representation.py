"""CoRP representation with an explicit valid-prefix mask."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Representation:
    tokens: torch.Tensor
    pooled_language: torch.Tensor
    action_dtype: torch.dtype
    mask: torch.Tensor | None = None
