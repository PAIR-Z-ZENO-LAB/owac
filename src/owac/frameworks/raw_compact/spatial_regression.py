"""A spatially resolved visual action estimate alongside the OWAC regression decoder."""

import torch
from torch import nn
from torch.nn import functional as F

from owac.frameworks.raw_compact.regression import RegressionPolicy


class SpatialActionRegression(RegressionPolicy):
    def __init__(self, cfg, *, initialize=False):
        super().__init__(cfg, initialize=initialize)
        width = cfg["model"]["raw_encoder"]["width"]
        self.compress = nn.Linear(width, 32)
        self.compress.requires_grad_(False)
        self.spatial_features = nn.Sequential(
            nn.Conv2d(32, 48, 3, padding=1),
            nn.GroupNorm(6, 48),
            nn.SiLU(),
            nn.Conv2d(48, 32, 3, padding=1),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
        )
        horizon = cfg["model"]["action_horizon"]
        dim = cfg["model"]["action_dim"]
        self.spatial_actions = nn.Sequential(
            nn.Linear(2 * 32 * 5 * 7, 384),
            nn.LayerNorm(384),
            nn.SiLU(),
            nn.Linear(384, 256),
            nn.SiLU(),
            nn.Linear(256, horizon * dim),
            nn.Unflatten(-1, (horizon, dim)),
        )
        self.base_logit = nn.Parameter(torch.zeros(dim))
        self.register_buffer("jaw_kernel", None, persistent=False)
        self.register_buffer("jaw_initial", None, persistent=False)
        self.register_buffer("jaw_index", None, persistent=False)
        if response := cfg["model"].get("aperture_response"):
            self.set_jaw_response(response["gain"], response["lead"])
        self.aperture = None
        if cfg["model"].get("conditional_aperture"):
            self.aperture = nn.Sequential(
                nn.Linear(2 * 32 * 3 * 4 + 7 + horizon * 7, 256),
                nn.LayerNorm(256),
                nn.SiLU(),
                nn.Linear(256, 128),
                nn.SiLU(),
                nn.Linear(128, horizon * 3),
                nn.Unflatten(-1, (horizon, 3)),
            )

    def spatial_map(self, patches, batch_size):
        h, w = self.cfg["observation"]["raw_shape"]
        side = self.cfg["model"]["raw_encoder"]["patch_size"]
        projected = (
            self.compress(patches)
            .reshape(2 * batch_size, h // side, w // side, -1)
            .permute(0, 3, 1, 2)
        )
        pooled = F.avg_pool2d(projected, 2)
        return pooled.unflatten(0, (2, batch_size)).transpose(0, 1)

    def visual_plan(self, maps):
        conv = self.spatial_features(maps.flatten(0, 1))
        pooled = F.adaptive_avg_pool2d(conv, (5, 7)).reshape(len(maps), -1)
        return self.spatial_actions(pooled)

    def set_jaw_response(self, gain, lead):
        horizon = self.cfg["model"]["action_horizon"]
        ticks = torch.arange(horizon, device=self.base_logit.device)
        offset = ticks[:, None] - ticks[None, :]
        self.jaw_kernel = torch.where(
            offset >= 0, gain * (1 - gain) ** offset.clamp_min(0), 0
        ).float()
        self.jaw_initial = ((1 - gain) ** (ticks + 1)).float()
        self.jaw_index = (ticks + lead).clamp_max(horizon - 1)

    def aperture_prediction(self, maps, state, vision):
        # The visual plan carries the estimated arm motion. The actual arm angles help
        # disambiguate grasp from release without feeding the previous jaw command back.
        scene = F.adaptive_avg_pool2d(maps.flatten(0, 1), (3, 4)).reshape(len(maps), -1)
        descriptor = torch.cat(
            (scene.float(), state[:, :7], vision[:, :, :7].flatten(1).float()), 1
        )
        return self.aperture(descriptor).float()

    def combine(self, base, vision, *, maps=None, state=None):
        action = torch.lerp(vision.float(), base.float(), self.base_logit.sigmoid())
        if self.aperture is not None:
            mode, closed, opened = self.aperture_prediction(maps, state, vision).unbind(-1)
            width = torch.where(mode > 0, opened, closed)
            action = torch.cat((action[:, :, :7], width.unsqueeze(-1)), -1)
        if self.jaw_kernel is not None:
            stats = self.cfg["statistics"]
            low, high = stats["action_q01"][7], stats["action_q99"][7]
            measured = state[:, 7] * (stats["state_std"][7] + 1e-6) + stats["state_mean"][7]
            initial = (measured - low) * (2 / (high - low + 1e-6)) - 1
            # Apply the fitted first-order response to successive predictions in this chunk.
            # The measured opening supplies the causal initial condition after each replan.
            with torch.autocast(action.device.type, enabled=False):
                width = action[:, :, 7].index_select(1, self.jaw_index) @ self.jaw_kernel.T
                width = width + initial[:, None] * self.jaw_initial
            action = torch.cat((action[:, :, :7], width[:, :, None]), -1)
        return action

    def predict(self, batch, *, return_goal=False):
        if return_goal:
            raise ValueError("This policy predicts spatial actions directly")
        rep, visual, patches = self.encode(batch, return_patches=True)
        base = self.action_generator(self.action_generator.prepare_prefix(rep))
        maps = self.spatial_map(patches, len(visual))
        return self.combine(base, self.visual_plan(maps), maps=maps, state=batch["state"]), visual
