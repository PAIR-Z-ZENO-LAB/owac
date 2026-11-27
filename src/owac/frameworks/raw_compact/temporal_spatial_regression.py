"""Condition a spatial OWAC action plan on recently measured arm and jaw motion."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from owac.frameworks.raw_compact.spatial_regression import SpatialActionRegression
from owac.runtime.teleop_history import causal_history_indices


class TemporalSpatialRegression(SpatialActionRegression):
    def __init__(self, cfg, *, initialize=False):
        super().__init__(cfg, initialize=initialize)
        h = cfg["model"]["action_horizon"]
        length = len(cfg["model"]["state_history_lags"])
        self.motion_correction = nn.Sequential(
            nn.Linear(2 * 32 * 3 * 4 + 8 * length + 8 + 2 * h * 8, 384),
            nn.LayerNorm(384),
            nn.SiLU(),
            nn.Linear(384, 256),
            nn.SiLU(),
            nn.Linear(256, h * 8),
            nn.Unflatten(-1, (h, 8)),
        )
        nn.init.zeros_(self.motion_correction[-2].weight)
        nn.init.zeros_(self.motion_correction[-2].bias)
        self.register_buffer("jaw_fit", None, persistent=False)
        if fit := cfg["model"].get("jaw_mode_fit"):
            self.set_jaw_fit(fit)

    def set_jaw_fit(self, fit):
        self.cfg["model"]["jaw_mode_fit"] = fit
        self.jaw_fit = self.base_logit.new_tensor(fit)

    def filtered_width(self, action, state, history):
        f = self.jaw_fit
        cfg = self.cfg["statistics"]
        half = (cfg["action_q99"][7] - cfg["action_q01"][7] + 1e-6) / 2
        raw = (action[:, :, 7] + 1) * half + cfg["action_q01"][7]
        q = torch.cat((history.flip(1)[:, :, 7], state[:, 7:8]), 1)
        q = q * (cfg["state_std"][7] + 1e-6) + cfg["state_mean"][7]
        transition = f[0]
        stationary = transition[0] / transition.sum()
        stay = 1 - transition.sum()

        def update(prob, x, dt, mu, var):
            prior = stationary + (prob - stationary) * stay**dt
            ratio = (
                (x - mu[0]).square() / var[0]
                - (x - mu[1]).square() / var[1]
                + torch.log(var[0] / var[1])
            ) / 2
            return torch.sigmoid(torch.logit(prior, eps=torch.finfo(raw.dtype).eps) + ratio)

        lags = [*reversed(self.cfg["model"]["state_history_lags"]), 0]
        posterior = stationary.expand(len(q))
        for i, lag in enumerate(lags):
            dt = 1 if not i else lags[i - 1] - lag
            posterior = update(posterior, q[:, i], dt, f[1], f[2])
        result = []
        for x in raw.unbind(1):
            posterior = update(posterior, x, 1, f[3], f[4])
            y = (x[:, None] * f[5] + f[6]).clamp(f[7], f[8])
            result.append(torch.lerp(y[:, 0], y[:, 1], posterior))
        width = (torch.stack(result, 1) - cfg["action_q01"][7]) / half - 1
        return torch.cat((action[:, :, :7], width[:, :, None]), -1)

    def correct(self, base, vision, maps, state, history):
        plan = self.combine(base, vision, maps=maps, state=state)
        if history is None:
            history = state[:, None, :].expand(-1, len(self.cfg["model"]["state_history_lags"]), -1)
        scene = F.adaptive_avg_pool2d(maps.flatten(0, 1), (3, 4)).reshape(len(maps), -1)
        x = torch.cat(
            (
                scene.float(),
                state,
                (state[:, None] - history).flatten(1),
                base.float().flatten(1),
                vision.float().flatten(1),
            ),
            1,
        )
        action = plan + self.motion_correction(x).float()
        if self.jaw_fit is not None:
            action = self.filtered_width(action, state, history)
        return action

    def predict(self, batch, *, return_goal=False, return_scene=False):
        if return_goal:
            raise ValueError("This policy predicts spatial actions directly")
        rep, visual, patches = self.encode(batch, return_patches=True)
        base = self.action_generator(self.action_generator.prepare_prefix(rep))
        maps = self.spatial_map(patches, len(visual))
        action = self.correct(
            base, self.visual_plan(maps), maps, batch["state"], batch.get("state_history")
        )
        if return_scene:
            visual = F.adaptive_avg_pool2d(maps.float().flatten(0, 1), (3, 4)).reshape(
                len(maps), -1
            )
        return action, visual

    def predict_live(self, history, unpack, *, return_scene=False):
        from owac.training.teleop_policy_eval import predict_actions

        obs = history[-1]
        lags = self.cfg["model"]["state_history_lags"]
        hz = self.cfg["observation"]["frequency_hz"]
        ix = causal_history_indices([item.at for item in history], lags, 1 / hz)[-1]
        past = np.stack([history[j].feedback.values for j in ix])
        raw = np.stack([unpack(f.data, f.width, f.height, f.stride) for f in obs.frames])
        if return_scene:
            action, scene = predict_actions(
                self, raw, obs.feedback.values, history=past[None], return_scene=True
            )
            return action[0], scene[0]
        return predict_actions(self, raw, obs.feedback.values, history=past[None])[0]
