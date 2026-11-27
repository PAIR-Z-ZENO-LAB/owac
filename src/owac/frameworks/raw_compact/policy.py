"""RAW-native CoRP adaptation, trained jointly without external network weights."""

import torch
from torch import nn
from torch.nn import functional as F

from owac.frameworks.raw_compact._vendor.action_generator import DiTActionGenerator
from owac.frameworks.raw_compact._vendor.cross_attention_resampler import CrossAttentionResampler
from owac.frameworks.raw_compact._vendor.representation import Representation
from owac.frameworks.raw_compact._vendor.state_encoder import StateEncoder
from owac.frameworks.raw_compact.conditioning import ConditioningRefiner
from owac.frameworks.raw_compact.goal_grounding import SpatialGoalHead, spatial_goal_loss
from owac.frameworks.raw_compact.spatial import inherit_policy, resize_raw_views
from owac.frameworks.raw_compact.task_mlp import specialize_action_mlps
from owac.training.compact_observation import observation_spec
from owac.training.compact_retention import feature_distance
from owac.training.raw_ssl import RawEncoder
from owac.training.raw_visual_initialization import initialize_visual


def masked_flow_loss(prediction, target, valid, *, group_size=None):
    """Retain logical minibatch weighting when physical batches are enlarged."""
    error = (prediction.float() - target.float()).square().mean(-1)
    if group_size is None:
        return (error * valid).sum() / valid.sum()
    weighted = (error * valid).reshape(-1, group_size, error.shape[-1])
    counts = valid.reshape_as(weighted).sum(dim=(1, 2))
    return (weighted.sum(dim=(1, 2)) / counts).mean()


def paired_flow_objective(velocity, paired, target, valid, *, group_size=None):
    """Coupled flow samples expose action variation caused only by sensor input.

    The average two-view regression equals squared mean error plus variance.
    Return that regression and its variance separately; the recipe explicitly
    chooses the additional variance weight. Padding and logical groups follow
    the original action objective, including samples with no exposure pair.
    """
    observed = masked_flow_loss(velocity, target, valid, group_size=group_size)
    altered = masked_flow_loss(paired, target, valid, group_size=group_size)
    variance = masked_flow_loss(velocity, paired, valid, group_size=group_size) / 4
    return (observed + altered) / 2, observed, altered, variance


class CompactPolicy(nn.Module):
    def __init__(self, cfg, *, initialize=False):
        super().__init__()
        from transformers import T5Config, T5EncoderModel

        self.cfg = cfg
        model = cfg["model"]
        self.freeze_raw_encoder = model.get("freeze_raw_encoder", False)
        self.view_names = list(observation_spec(cfg)["views"])
        self.raw_encoder = RawEncoder(model["raw_encoder"])
        visual_metadata = None
        if initialize and model["visual_initialization"] in (
            "j5",
            "owned_policy_encoder",
            "owned_raw_encoder",
        ):
            visual_metadata = initialize_visual(self.raw_encoder, cfg)
        backbone = self.raw_encoder.backbone
        # No masked-image objective is used by this action policy.
        backbone.embeddings.mask_token.requires_grad_(False)
        backbone.config.pos_embed_shift = None
        backbone.config.pos_embed_jitter = None
        backbone.config.pos_embed_rescale = None
        for block in backbone.layer:
            block.drop_path = nn.Identity()
        if self.freeze_raw_encoder:
            self.raw_encoder.requires_grad_(False).eval()
        language = T5Config(**model["language"])
        self.language_encoder = T5EncoderModel(language)
        self.language_projection = nn.Sequential(
            nn.LayerNorm(language.d_model), nn.Linear(language.d_model, model["hidden_dim"])
        )
        self.resampler = CrossAttentionResampler(
            view_names=self.view_names,
            hidden_dim=model["hidden_dim"],
            tokens_per_view=model["tokens_per_view"],
            num_heads=model["resampler_heads"],
            film_rank=model["film_rank"],
            input_dim=model["raw_encoder"]["width"],
            spatial_coordinates=model.get("spatial_coordinates", False),
        )
        self.state_encoder = StateEncoder(model.get("state_dim", 9), model["hidden_dim"])
        self.action_generator = DiTActionGenerator(model)
        self.task_head = nn.Linear(model["hidden_dim"], len(cfg["task_names"]))
        # Construct optional new parameters after every legacy module. Whole-policy
        # inheritance keeps their zero-gated initialization unless they were saved.
        self.prefix_refinement = ConditioningRefiner(model)
        goal = model.get("spatial_goal_head")
        self.spatial_goal_head = (
            SpatialGoalHead(
                model["raw_encoder"]["width"], model["hidden_dim"], goal["width"], goal["arms"]
            )
            if goal
            else None
        )
        self.task_action_mlp = model.get("task_action_mlp", False)
        if self.task_action_mlp:
            specialize_action_mlps(self.action_generator, len(cfg["task_names"]))
        self.initialization = visual_metadata
        if initialize and model["visual_initialization"] == "policy":
            payload = torch.load(
                cfg["assets"]["policy"], map_location="cpu", mmap=True, weights_only=False
            )
            if observation_spec(payload["config"])["domain"] != observation_spec(cfg)["domain"]:
                raise ValueError("Owned policy must come from the same platform")
            self.initialization = inherit_policy(
                self, payload, patch_resize=model.get("policy_patch_resize")
            )

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_raw_encoder:
            self.raw_encoder.eval()
        return self

    def conditioning(self, batch):
        """Share text dropout and proprioception across counterfactual RAW views."""
        language = self.language_projection(
            self.language_encoder(
                input_ids=batch["tokens"], attention_mask=batch["token_mask"]
            ).last_hidden_state
        )
        language = language * batch["token_mask"].unsqueeze(-1)
        pooled = language.sum(1) / batch["token_mask"].sum(1, keepdim=True)
        return language, pooled, self.state_encoder(batch["state"])

    def encode_raw(self, raw):
        raw = resize_raw_views(
            raw,
            self.cfg["model"].get("raw_image_size"),
            sensor=getattr(self.raw_encoder.response, "sensor", None),
        )
        _, tokens = self.raw_encoder(raw.transpose(0, 1).flatten(0, 1))
        return raw, tokens

    def encode_observation(self, batch):
        """Visual hook for encoders that also consume observation metadata."""
        return self.encode_raw(batch["raw"])

    def encode(self, batch, *, return_patches=False, context=None):
        raw, tokens = self.encode_observation(batch)
        patch = self.cfg["model"]["raw_encoder"]["patch_size"]
        grid = (raw.shape[-2] // patch, raw.shape[-1] // patch)
        return self.fuse_patches(
            batch, tokens, grid, return_patches=return_patches, context=context
        )

    def fuse_patches(self, batch, tokens, grid, *, return_patches=False, context=None):
        size = batch["state"].shape[0]
        language, pooled, state = self.conditioning(batch) if context is None else context
        patches = dict(zip(self.view_names, tokens.split(size), strict=True))
        visual = self.resampler(patches, pooled, grid)
        representation = self.condition_visual(batch, visual, language, pooled, state)
        result = representation, visual
        return (*result, tokens) if return_patches else result

    def condition_visual(self, batch, visual, language, pooled, state):
        """Combine resampled camera tokens with the current language and measured state."""
        size = len(visual)
        prefix = torch.cat((visual, language, state), 1)
        # Training may omit the whole proprioceptive modality for selected
        # samples. Mask its attention key instead of inventing a sensor value.
        state_mask = batch.get("state_visible")
        state_mask = (
            torch.ones((size, 1), dtype=torch.bool, device=visual.device)
            if state_mask is None
            else state_mask[:, None]
        )
        mask = torch.cat(
            (
                torch.ones(visual.shape[:2], dtype=torch.bool, device=visual.device),
                batch["token_mask"],
                state_mask,
            ),
            1,
        )
        prefix = self.prefix_refinement(prefix, mask)
        return Representation(prefix, pooled, visual.dtype, mask)

    def forward(
        self,
        batch,
        *,
        noise=None,
        tau=None,
        return_retention=False,
        return_spatial_goal=False,
        return_exposure=False,
    ):
        actions = batch["actions"]
        noise = torch.randn_like(actions) if noise is None else noise
        if tau is None:
            tau = torch.distributions.Beta(1.0, 1.5).sample((len(actions),)).to(actions.device)
        noisy = noise.lerp(actions, tau[:, None, None])
        context = self.conditioning(batch) if return_exposure else None
        representation, visual, patches = self.encode(batch, return_patches=True, context=context)
        caches = self.action_generator.prepare_prefix(
            representation, batch["task_id"] if self.task_action_mlp else None
        )
        velocity = self.action_generator(noisy, tau, caches)
        group_size = self.cfg["training"].get("loss_group_size")
        flow = masked_flow_loss(
            velocity, actions - noise, batch["action_valid"], group_size=group_size
        )
        task = F.cross_entropy(self.task_head(visual.mean(1)).float(), batch["task_id"])
        loss = flow + self.cfg["training"]["task_loss_weight"] * task
        extra = []
        if return_retention:
            degraded_raw = resize_raw_views(
                batch["degraded_raw"],
                self.cfg["model"].get("raw_image_size"),
                sensor=getattr(self.raw_encoder.response, "sensor", None),
            )
            _, degraded = self.raw_encoder(degraded_raw.transpose(0, 1).flatten(0, 1))
            anchor = feature_distance(patches, batch["teacher_patches"])
            exposure = feature_distance(degraded, batch["teacher_patches"])
            loss = loss + self.cfg["visual_retention"]["weight"] * (anchor + exposure) / 2
            extra.append(torch.stack((anchor, exposure)).detach())
        if self.spatial_goal_head is not None:
            # An extra CURRENT, unaugmented external RAW view avoids inventing a
            # geometry transform for the legacy action-image augmentation. This
            # costs one additional visual forward/backward in training only.
            goal_raw = resize_raw_views(
                batch["goal_raw"][:, None],
                self.cfg["model"].get("raw_image_size"),
                sensor=getattr(self.raw_encoder.response, "sensor", None),
            )[:, 0]
            _, goal_patches = self.raw_encoder(goal_raw)
            logits = self.spatial_goal_head(goal_patches, representation.pooled_language)
            patch = self.cfg["model"]["raw_encoder"]["patch_size"]
            height, width = goal_raw.shape[-2:]
            grid = (height // patch, width // patch)
            uv = batch["spatial_goal_uv"] * batch["spatial_goal_uv"].new_tensor(
                [width / (grid[1] * patch), height / (grid[0] * patch)]
            )
            visible = batch["spatial_goal_valid"] & ((uv >= 0) & (uv < 1)).all(-1)
            goal_loss, _ = spatial_goal_loss(logits, uv, visible, grid, group_size=group_size)
            loss = loss + self.cfg["spatial_goals"]["loss_weight"] * goal_loss
            if return_spatial_goal:
                extra.append(torch.stack((goal_loss.detach(), visible.float().mean())))
        if return_exposure:
            indices = batch["exposure_pair_valid"].nonzero().flatten()
            paired = velocity
            if len(indices):
                alternate = {key: value[indices] for key, value in batch.items()}
                alternate["raw"] = alternate["degraded_raw"]
                paired_representation, _ = self.encode(
                    alternate, context=tuple(value[indices] for value in context)
                )
                paired_caches = self.action_generator.prepare_prefix(
                    paired_representation, alternate["task_id"] if self.task_action_mlp else None
                )
                # Identical noise, interpolation time, language realization and
                # state. Differences originate in the RAW observation path.
                paired = velocity.index_copy(
                    0, indices, self.action_generator(noisy[indices], tau[indices], paired_caches)
                )
            paired_flow, observed, altered, variance = paired_flow_objective(
                velocity, paired, actions - noise, batch["action_valid"], group_size=group_size
            )
            loss = (
                loss
                - flow
                + paired_flow
                + self.cfg["action_exposure"]["variance_weight"] * variance
            )
            flow = paired_flow
            extra.append(
                torch.stack(
                    (
                        observed.detach(),
                        altered.detach(),
                        variance.detach(),
                        batch["exposure_pair_valid"].float().mean(),
                    )
                )
            )
        return loss, flow.detach(), task.detach(), *extra

    @torch.no_grad()
    def sample(self, batch, noise, steps=None):
        steps = steps or self.cfg["model"]["action_generator"]["num_sample_steps"]
        representation, _ = self.encode(batch)
        caches = self.action_generator.prepare_prefix(
            representation, batch["task_id"] if self.task_action_mlp else None
        )
        actions = noise.float()
        for index in range(steps):
            tau = actions.new_full((len(actions),), index / steps)
            actions = actions + self.action_generator(actions, tau, caches).float() / steps
        return actions
