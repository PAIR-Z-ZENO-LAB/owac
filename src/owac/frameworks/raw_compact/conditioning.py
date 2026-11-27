"""Joint observation refinement before the existing CoRP action caches.

Zero-initialized residual gates preserve the inherited policy at initialization.
This is an observation-only Transformer, not an additional flow integration step.
"""

import torch
from torch import nn

from owac.frameworks.raw_compact._vendor.action_generator import SelfAttention


class ConditioningBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_ratio):
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = SelfAttention(hidden_dim, num_heads)
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
        )
        self.attention_gate = nn.Parameter(torch.zeros(hidden_dim))
        self.mlp_gate = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, tokens, mask):
        query, key, value = self.attention.project_qkv(self.attention_norm(tokens))
        update = self.attention.attend(query, key, value, mask)
        # Keep the prefix's dtype under autocast, including at the identity start.
        tokens = tokens + self.attention_gate.to(tokens.dtype) * update
        return tokens + self.mlp_gate.to(tokens.dtype) * self.mlp(self.mlp_norm(tokens))


class ConditioningRefiner(nn.Module):
    def __init__(self, model):
        super().__init__()
        layers = model.get("prefix_refinement_layers", 0)
        if layers < 0:
            raise ValueError("prefix_refinement_layers must be nonnegative")
        generator = model["action_generator"]
        self.blocks = nn.ModuleList(
            ConditioningBlock(model["hidden_dim"], generator["num_heads"], generator["mlp_ratio"])
            for _ in range(layers)
        )

    def forward(self, tokens, valid):
        mask = valid[:, None, None, :]
        for block in self.blocks:
            tokens = block(tokens, mask)
        return tokens
