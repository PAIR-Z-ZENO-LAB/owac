"""One-pass action-query regression using the reference DiT's prefix attention."""

import math

import torch
from torch import nn

from .action_generator import DiTActionGenerator


class RegressionActionGenerator(nn.Module):
    def __init__(self, reference: DiTActionGenerator):
        super().__init__()
        self.action_dim = reference.action_dim
        self.action_horizon = reference.action_horizon
        self.blocks = reference.blocks
        self.output_norm = reference.output_norm
        self.output_projection = reference.velocity_projection
        # Consume the original DiT initialization first; queries cannot shift later modules' RNG.
        with torch.random.fork_rng(devices=[]):
            self.queries = nn.Parameter(
                torch.randn(1, self.action_horizon, self.output_norm.normalized_shape[0])
                / math.sqrt(self.output_norm.normalized_shape[0])
            )

    prepare_prefix = DiTActionGenerator.prepare_prefix

    def forward(self, prefix):
        tokens = self.queries.expand(prefix[0].key.shape[0], -1, -1)
        for block, cache in zip(self.blocks, prefix, strict=True):
            tokens = block(tokens, cache)
        return self.output_projection(self.output_norm(tokens))
