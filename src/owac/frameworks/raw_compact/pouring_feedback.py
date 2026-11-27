"""Pouring action correction using two native RAW grids and recent measured motion."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from owac.frameworks.raw_compact.regression import RegressionPolicy
from owac.runtime.teleop_history import causal_history_indices


def stereo_grid(patches, cfg, size):
    height, width = cfg["observation"]["raw_shape"]
    patch = cfg["model"]["raw_encoder"]["patch_size"]
    maps = patches.transpose(1, 2).reshape(2 * size, -1, height // patch, width // patch)
    grid = F.adaptive_avg_pool2d(maps.float(), (3, 4)).flatten(2).transpose(1, 2)
    return grid.reshape(2, size, 12, -1).transpose(0, 1).flatten(1, 2)


class PouringFeedbackPolicy(RegressionPolicy):
    def __init__(self, cfg, *, initialize=False):
        super().__init__(cfg, initialize=initialize)
        width = cfg["model"]["raw_encoder"]["width"]
        steps = cfg["model"]["action_horizon"]
        length = len(cfg["model"]["state_history_lags"])
        self.register_buffer("grid_center", torch.zeros(24, width))
        self.register_buffer("grid_scale", torch.ones(24, width))
        self.grid_readout = nn.Sequential(nn.Linear(width, 32), nn.LayerNorm(32), nn.SiLU())
        self.feedback_head = nn.Sequential(
            nn.Linear(24 * 32 + 8 * (length + 1) + steps * 8, 384),
            nn.LayerNorm(384),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, steps * 8),
            nn.Unflatten(-1, (steps, 8)),
        )
        nn.init.zeros_(self.feedback_head[-2].weight)
        nn.init.zeros_(self.feedback_head[-2].bias)
        if cfg["model"].get("pouring_absolute_anchor"):
            self.anchor_grid = nn.Sequential(nn.Linear(width, 48), nn.LayerNorm(48), nn.SiLU())
            self.anchor_head = nn.Sequential(
                nn.Linear(24 * 48, 384),
                nn.LayerNorm(384),
                nn.SiLU(),
                nn.Dropout(0.10),
                nn.Linear(384, 256),
                nn.LayerNorm(256),
                nn.SiLU(),
                nn.Linear(256, steps * 8),
                nn.Unflatten(-1, (steps, 8)),
            )
            self.trust_logit = nn.Parameter(torch.zeros(steps, 8))
            self.disagreement = nn.Parameter(torch.zeros(8))
            stats = cfg["statistics"]
            for name, value in (
                ("q_center", stats["state_mean"]),
                ("q_scale", np.array(stats["state_std"]) + 1e-6),
                ("low", stats["action_q01"]),
                (
                    "action_half",
                    (np.array(stats["action_q99"]) - np.array(stats["action_q01"]) + 1e-6) / 2,
                ),
                ("relative", cfg["observation"]["relative_action_mask"]),
            ):
                self.register_buffer(
                    name, torch.tensor(value, dtype=torch.float32), persistent=False
                )

    def feedback(self, base, grid, state, history):
        scene = self.grid_readout((grid.float() - self.grid_center) / self.grid_scale)
        inputs = torch.cat(
            (scene.flatten(1), state, (state[:, None] - history).flatten(1), base.flatten(1)), 1
        )
        return base.float() + self.feedback_head(inputs).float()

    def absolute(self, grid):
        normalized = (grid.float() - self.grid_center) / self.grid_scale
        scene = self.anchor_grid(normalized).flatten(1)
        return self.anchor_head(scene).float() * self.q_scale + self.q_center

    def blend(self, previous, anchor, state):
        origin = ((state * self.q_scale + self.q_center) * self.relative)[:, None]
        absolute = (previous + 1) * self.action_half + self.low + origin
        distance = (absolute - anchor).abs() / self.action_half
        trust = (self.trust_logit - F.softplus(self.disagreement) * distance).sigmoid()
        return (torch.lerp(anchor, absolute, trust) - origin - self.low) / self.action_half - 1

    def correct(self, base, grid, state, history):
        previous = self.feedback(base, grid, state, history)
        if hasattr(self, "anchor_head"):
            return self.blend(previous, self.absolute(grid), state)
        return previous

    def predict(self, batch, *, return_goal=False):
        if return_goal:
            raise ValueError("Pouring feedback predicts actions")
        rep, visual, patches = self.encode(batch, return_patches=True)
        base = self.action_generator(self.action_generator.prepare_prefix(rep))
        state = batch["state"]
        length = len(self.cfg["model"]["state_history_lags"])
        past = batch.get("state_history", state[:, None].expand(-1, length, -1))
        return self.correct(base, stereo_grid(patches, self.cfg, len(state)), state, past), visual

    def predict_live(self, history, unpack):
        from owac.training.teleop_policy_eval import predict_actions

        current = history[-1]
        cfg = self.cfg
        indices = causal_history_indices(
            [item.at for item in history],
            cfg["model"]["state_history_lags"],
            1 / cfg["observation"]["frequency_hz"],
        )[-1]
        past = np.stack([history[i].feedback.values for i in indices])
        raw = np.stack([unpack(f.data, f.width, f.height, f.stride) for f in current.frames])
        return predict_actions(self, raw, current.feedback.values, history=past[None])[0]
