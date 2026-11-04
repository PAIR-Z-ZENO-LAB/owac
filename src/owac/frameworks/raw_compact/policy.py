"""RAW-native CoRP adaptation, trained jointly without external network weights."""

import torch
from torch import nn
from torch.nn import functional as F

from owac.frameworks.raw_compact._vendor.action_generator import DiTActionGenerator
from owac.frameworks.raw_compact._vendor.cross_attention_resampler import CrossAttentionResampler
from owac.frameworks.raw_compact._vendor.representation import Representation
from owac.frameworks.raw_compact._vendor.state_encoder import StateEncoder
from owac.training.raw_ssl import RawEncoder


def masked_flow_loss(prediction, target, valid):
    error = (prediction.float() - target.float()).square().mean(-1)
    return (error * valid).sum() / valid.sum()


class CompactPolicy(nn.Module):
    def __init__(self, cfg, *, initialize=False):
        super().__init__()
        from transformers import T5Config, T5EncoderModel

        self.cfg = cfg
        model = cfg["model"]
        self.raw_encoder = RawEncoder(model["raw_encoder"])
        if initialize and model["visual_initialization"] == "j5":
            payload = torch.load(
                cfg["assets"]["j5"], map_location="cpu", mmap=True, weights_only=False
            )
            self.raw_encoder.load_state_dict(payload["encoder"])
        backbone = self.raw_encoder.backbone
        # No masked-image objective is used by this action policy.
        backbone.embeddings.mask_token.requires_grad_(False)
        backbone.config.pos_embed_shift = None
        backbone.config.pos_embed_jitter = None
        backbone.config.pos_embed_rescale = None
        for block in backbone.layer:
            block.drop_path = nn.Identity()
        language = T5Config(**model["language"])
        self.language_encoder = T5EncoderModel(language)
        self.language_projection = nn.Sequential(
            nn.LayerNorm(language.d_model), nn.Linear(language.d_model, model["hidden_dim"])
        )
        self.resampler = CrossAttentionResampler(
            view_names=["agentview", "eye_in_hand"],
            hidden_dim=model["hidden_dim"],
            tokens_per_view=model["tokens_per_view"],
            num_heads=model["resampler_heads"],
            film_rank=model["film_rank"],
        )
        self.state_encoder = StateEncoder(9, model["hidden_dim"])
        self.action_generator = DiTActionGenerator(model)
        self.task_head = nn.Linear(model["hidden_dim"], len(cfg["task_names"]))

    def encode(self, batch):
        raw = batch["raw"]
        size = raw.shape[0]
        _, tokens = self.raw_encoder(torch.cat((raw[:, 0], raw[:, 1]), 0))
        language = self.language_projection(
            self.language_encoder(
                input_ids=batch["tokens"], attention_mask=batch["token_mask"]
            ).last_hidden_state
        )
        language = language * batch["token_mask"].unsqueeze(-1)
        pooled = language.sum(1) / batch["token_mask"].sum(1, keepdim=True)
        visual = self.resampler({"agentview": tokens[:size], "eye_in_hand": tokens[size:]}, pooled)
        state = self.state_encoder(batch["state"])
        prefix = torch.cat((visual, language, state), 1)
        mask = torch.cat(
            (
                torch.ones(visual.shape[:2], dtype=torch.bool, device=raw.device),
                batch["token_mask"],
                torch.ones((size, 1), dtype=torch.bool, device=raw.device),
            ),
            1,
        )
        return Representation(prefix, pooled, visual.dtype, mask), visual

    def forward(self, batch, *, noise=None, tau=None):
        actions = batch["actions"]
        noise = torch.randn_like(actions) if noise is None else noise
        if tau is None:
            tau = torch.distributions.Beta(1.0, 1.5).sample((len(actions),)).to(actions.device)
        noisy = noise.lerp(actions, tau[:, None, None])
        representation, visual = self.encode(batch)
        caches = self.action_generator.prepare_prefix(representation)
        velocity = self.action_generator(noisy, tau, caches)
        flow = masked_flow_loss(velocity, actions - noise, batch["action_valid"])
        task = F.cross_entropy(self.task_head(visual.mean(1)).float(), batch["task_id"])
        return flow + self.cfg["training"]["task_loss_weight"] * task, flow.detach(), task.detach()

    @torch.no_grad()
    def sample(self, batch, noise, steps=None):
        steps = steps or self.cfg["model"]["action_generator"]["num_sample_steps"]
        representation, _ = self.encode(batch)
        caches = self.action_generator.prepare_prefix(representation)
        actions = noise.float()
        for index in range(steps):
            tau = actions.new_full((len(actions),), index / steps)
            actions = actions + self.action_generator(actions, tau, caches).float() / steps
        return actions
