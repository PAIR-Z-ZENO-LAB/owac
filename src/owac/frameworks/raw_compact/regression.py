"""OWAC C1 learned-query regression with the current native RAW encoder."""

import math
from dataclasses import replace

import torch
from torch.nn import functional as F

from owac.frameworks.raw_compact._vendor.regression_action_generator import (
    RegressionActionGenerator,
)
from owac.frameworks.raw_compact.policy import CompactPolicy


class RegressionPolicy(CompactPolicy):
    def __init__(self, cfg, *, initialize=False):
        super().__init__(cfg, initialize=initialize)
        self.action_generator = RegressionActionGenerator(self.action_generator)
        self.banana_projection = None
        if cfg["model"].get("banana_grounding"):
            self.banana_projection = torch.nn.Linear(5, cfg["model"]["hidden_dim"], bias=False)
            torch.nn.init.zeros_(self.banana_projection.weight)
        if initialize and cfg["model"].get("language_initialization") == "pretrained":
            from transformers import T5EncoderModel

            with torch.random.fork_rng(devices=[]):
                pretrained, info = T5EncoderModel.from_pretrained(
                    cfg["assets"]["language_encoder"],
                    config=self.language_encoder.config,
                    local_files_only=True,
                    output_loading_info=True,
                )
            if info["missing_keys"] or info["mismatched_keys"] or info["error_msgs"]:
                raise ValueError(f"Incomplete T5 encoder initialization: {info}")
            self.language_encoder.load_state_dict(pretrained.state_dict(), strict=True)

    def predict(self, batch, *, return_goal=False):
        grounded = self.banana_projection is not None
        if return_goal and not grounded:
            raise ValueError("This regression policy has no banana location head")
        if grounded or "features" in batch:
            height, width = self.cfg["observation"]["raw_shape"]
            patch = self.cfg["model"]["raw_encoder"]["patch_size"]
            grid = height // patch, width // patch
        if "features" in batch:
            patches = batch["features"].transpose(0, 1).flatten(0, 1)
            encoded = self.fuse_patches(batch, patches, grid, return_patches=grounded)
        else:
            encoded = self.encode(batch, return_patches=grounded)
        representation, visual = encoded[:2]
        if grounded:
            u2 = encoded[2][: len(batch["state"])]
            logits = self.spatial_goal_head(u2, representation.pooled_language)[:, 0]
            y, x = torch.meshgrid(
                (torch.arange(grid[0], device=u2.device) + 0.5) * 2 / grid[0] - 1,
                (torch.arange(grid[1], device=u2.device) + 0.5) * 2 / grid[1] - 1,
                indexing="ij",
            )
            coords = torch.stack((x, y, x * x, x * y, y * y), -1).reshape(-1, 5)
            shape = logits.float().softmax(-1) @ coords
            shift = self.banana_projection(shape)[:, None]
            tokens = representation.tokens
            representation = replace(
                representation, tokens=torch.cat((tokens[:, :1] + shift, tokens[:, 1:]), 1)
            )
        cache = self.action_generator.prepare_prefix(representation)
        result = self.action_generator(cache), visual
        return (*result, logits) if return_goal else result

    def forward(self, batch, *, noise=None, tau=None, return_prediction=False):
        goal = self.banana_projection is not None and return_prediction and "banana_mass" in batch
        result = self.predict(batch, return_goal=goal)
        prediction, visual = result[:2]
        valid = batch["action_valid"]
        mse = F.mse_loss(prediction.float()[valid], batch["actions"].float()[valid])
        task = F.cross_entropy(self.task_head(visual.mean(1)).float(), batch["task_id"])
        losses = mse + self.cfg["training"]["task_loss_weight"] * task, mse.detach(), task.detach()
        if not return_prediction:
            return losses
        if goal:
            logits = result[2].float()
            ce = -(batch["banana_mass"] * logits.log_softmax(-1)).sum(-1)
            weights = batch["banana_visible"]
            supervised = (
                (ce * weights).sum() / weights.sum().clamp_min(1) / math.log(logits.shape[-1])
            )
            return *losses, prediction, supervised
        return *losses, prediction

    @torch.no_grad()
    def sample(self, batch, noise=None, steps=None):
        return self.predict(batch)[0].float()
