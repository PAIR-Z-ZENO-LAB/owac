"""Explicit plain-patch control for the archived J5/compact-policy recipes.

The control keeps only the original linear RAW channels. Random controls use a
fresh input projection; legacy continuation controls explicitly truncate an owned
patch kernel and do not claim a function-preserving conversion.
"""

import hashlib
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn


def plain_patch_state(state):
    result = {}
    for name, value in state.items():
        if name.endswith("response.log_pivot"):
            continue
        if name.endswith("backbone.embeddings.patch_embeddings.weight"):
            if value.shape[1] not in (3, 9):
                raise ValueError("Plain-patch conversion expects 3 or 9 input channels")
            value = value[:, :3].clone()
        result[name] = value
    return result


def cached_reader(original, directory):
    """Use atomically completed, byte-verified private copies as they arrive."""

    class CachedFrameReader(original):
        def __init__(self, path, camera):
            candidate = Path(directory) / (hashlib.sha256(str(path).encode()).hexdigest() + ".npz")
            super().__init__(candidate if candidate.is_file() else path, camera)

    return CachedFrameReader


def encoder_class(original):
    class AblationEncoder(original):
        def __init__(self, cfg):
            super().__init__(cfg)
            if cfg.get("input_encoding", "owac") == "plain_patch":
                previous = self.backbone.embeddings.patch_embeddings
                # Preserve the RNG stream used by all shared layers/SSL heads.
                # Random controls have their own normally initialized input layer;
                # legacy continuation controls retain their explicit truncation.
                random_input = cfg.get("patch_initialization", "truncate") == "random"
                with torch.random.fork_rng(devices=[]) if random_input else nullcontext():
                    projection = nn.Conv2d(
                        3,
                        previous.out_channels,
                        previous.kernel_size,
                        stride=previous.stride,
                        bias=previous.bias is not None,
                    )
                    if random_input:
                        nn.init.normal_(
                            projection.weight,
                            std=getattr(self.backbone.config, "initializer_range", 0.02),
                        )
                        if projection.bias is not None:
                            nn.init.zeros_(projection.bias)
                    else:
                        with torch.no_grad():
                            projection.weight.copy_(previous.weight[:, :3])
                            if previous.bias is not None:
                                projection.bias.copy_(previous.bias)
                self.backbone.embeddings.patch_embeddings = projection
                self.backbone.config.num_channels = 3
                self.response = nn.Identity()
            elif cfg.get("input_encoding", "owac") != "owac":
                raise ValueError("Unknown RAW encoding")
            self.frozen_readout = cfg.get("freeze_for_action_readout", False)
            if self.frozen_readout:
                self.requires_grad_(False)
                self.eval()

        def train(self, mode=True):
            # The archived trainer calls policy.train() after every validation.
            return super().train(False if self.frozen_readout else mode)

    return AblationEncoder
