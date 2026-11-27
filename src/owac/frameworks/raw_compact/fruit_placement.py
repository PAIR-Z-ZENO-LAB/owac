"""Select the pickup or transport OWAC regressor from the two observed views and arm state."""

import copy

import torch
from torch import nn
from torch.nn import functional as F

from owac.frameworks.raw_compact.regression import RegressionPolicy


def placement_descriptor(features, state, grid):
    """Keep the layout of the frozen patch features in each camera."""
    b, views, _, width = features.shape
    images = features.transpose(2, 3).reshape(b * views, width, *grid)
    pooled = F.adaptive_avg_pool2d(images.float(), (2, 3)).reshape(b, -1)
    return torch.cat((pooled, state.float()), dim=1)


class PlacementGate(nn.Module):
    def __init__(self, width):
        super().__init__()
        size = 2 * 2 * 3 * width + 8
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("scale", torch.ones(size))
        self.layers = nn.Sequential(nn.Linear(size, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, descriptor):
        return self.layers((descriptor - self.mean) / self.scale)


class ReleaseAperture(nn.Module):
    """Regress each future jaw width conditional on holding or opening the fingers."""

    def __init__(self, cfg, landmarks):
        super().__init__()
        width = cfg["model"]["raw_encoder"]["width"]
        horizon = cfg["model"]["action_horizon"]
        self.register_buffer("state_mean", torch.tensor(cfg["statistics"]["state_mean"]))
        self.register_buffer("state_std", torch.tensor(cfg["statistics"]["state_std"]) + 1e-6)
        for name, shape in (("xyz", (7, 3)), ("rot", (7, 3, 3)), ("tip", (3,)), ("basis", (3, 3))):
            self.register_buffer(name, torch.zeros(shape))
        self.register_buffer("releases", torch.zeros(landmarks, 3))
        dim = 13 + 2 * landmarks
        self.register_buffer("geom_mean", torch.zeros(dim))
        self.register_buffer("geom_scale", torch.ones(dim))
        self.visual = nn.Sequential(nn.Linear(width * 12 + 8, 24), nn.LayerNorm(24), nn.GELU())
        self.geometry = nn.Sequential(nn.Linear(dim, 64), nn.LayerNorm(64), nn.GELU())
        self.output = nn.Sequential(nn.Linear(88, 128), nn.GELU(), nn.Linear(128, horizon * 3))

    def geometry_features(self, state):
        from owac.training.fruit_grasp import nero_pose

        with torch.autocast(state.device.type, enabled=False):
            measured = state.float() * self.state_std + self.state_mean
            p, rot = nero_pose(measured, self.xyz, self.rot, self.tip, self.basis)
            delta = p[:, None] - self.releases
            return torch.cat(
                (
                    p,
                    rot.flatten(1),
                    measured[:, 7:],
                    delta[..., :2].norm(dim=-1),
                    delta.norm(dim=-1),
                ),
                dim=1,
            )

    def forward(self, descriptor, geometry):
        v = self.visual(descriptor)
        g = self.geometry((geometry - self.geom_mean) / self.geom_scale)
        return self.output(torch.cat((v, g), dim=1)).reshape(len(g), -1, 3).float().unbind(-1)

    def predict(self, descriptor, geometry):
        logit, holding, opening = self(descriptor, geometry)
        return torch.where(logit > 0, opening, holding)


class FruitPlacementPolicy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        core = copy.deepcopy(cfg)
        core["model"]["kind"] = "corp_regression"
        self.pickup = RegressionPolicy(core)
        self.carry = RegressionPolicy(core)
        # Both regression heads consume tokens from the same frozen physical-camera encoder.
        self.carry.raw_encoder = nn.Identity()
        encoder = core["model"]["raw_encoder"]
        self.grid = tuple(x // encoder["patch_size"] for x in cfg["observation"]["raw_shape"])
        self.gate = PlacementGate(encoder["width"])
        self.aperture = None
        if count := core["model"].get("release_landmarks"):
            self.aperture = ReleaseAperture(core, count)

    @torch.no_grad()
    def sample(self, batch, noise=None, steps=None):
        n = len(batch["state"])
        features = batch.get("features")
        if features is None:
            _, packed = self.pickup.encode_raw(batch["raw"])
            features = packed.reshape(2, n, *packed.shape[1:]).transpose(0, 1)
        descriptor = placement_descriptor(features, batch["state"], self.grid)
        which = self.gate(descriptor).argmax(-1)
        inputs = {k: v for k, v in batch.items() if k != "raw"}
        inputs["features"] = features
        actions = batch["state"].new_empty((n, self.cfg["model"]["action_horizon"], 8))
        for index, expert in enumerate((self.pickup, self.carry)):
            rows = (which == index).nonzero().flatten()
            if len(rows):
                selected = inputs if len(rows) == n else {k: v[rows] for k, v in inputs.items()}
                prediction = expert.sample(selected)
                if index == 1 and self.aperture is not None:
                    normal = (descriptor[rows] - self.gate.mean) / self.gate.scale
                    geom = self.aperture.geometry_features(selected["state"])
                    prediction[..., 7] = self.aperture.predict(normal, geom)
                if len(rows) == n:
                    return prediction
                actions[rows] = prediction
        return actions
