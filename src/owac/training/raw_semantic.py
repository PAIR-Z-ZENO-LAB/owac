"""Complete RAW-only dense encoder; RGB DINO supervision stays outside this module."""

import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from owac.baselines.raw_stem_siglip import LocalConvStem
from owac.training.raw_jepa import RawVisualEncoder


class RadiometricStem(nn.Module):
    """Linear measurement plus learned log-response features, not an RGB image."""

    def __init__(self, original):
        super().__init__()
        self.original = original
        self.log_conv = nn.Conv2d(original.in_channels, original.conv_channels, 3, padding=1)
        nn.init.zeros_(self.log_conv.weight)
        nn.init.zeros_(self.log_conv.bias)

    def forward(self, raw):
        # Current dataset is uint8 pseudo-RAW /255; log branch retains dark differences.
        # No invented sensor calibration and no RGB reconstruction target.
        log = torch.log1p(raw.float() * 255) / math.log(256)
        values = self.original.conv64(raw) + self.log_conv(log.to(raw.dtype))
        return self.original.patch_embedding(self.original.activation(values))


class RawSemanticEncoder(nn.Module):
    """Radiometric stem -> content-gated causal history -> full tower -> dense semantics."""

    def __init__(self, base, *, output_width=1024, depths=(9, 18, 27)):
        super().__init__()
        self.vision = base.vision
        self.gate_logit = base.gate_logit
        self.width = base.width
        self.depths = tuple(depths)
        self.checkpoint_blocks = True
        self.vision.embeddings.patch_embedding = RadiometricStem(
            self.vision.embeddings.patch_embedding
        )
        # Shared learned content gate per spatial site plus inherited channel prior.
        self.content_gate = nn.Linear(3 * self.width, 1)
        nn.init.zeros_(self.content_gate.weight)
        nn.init.zeros_(self.content_gate.bias)
        self.depth_logits = nn.Parameter(torch.tensor([2.0] + [0.0] * (len(depths) - 1)))
        self.semantic_projection = nn.Linear(self.width, output_width)

    def forward(self, raw):
        batch, frames = raw.shape[:2]
        tokens = self.vision.embeddings.patch_embedding(raw.flatten(0, 1))
        tokens = tokens.flatten(2).transpose(1, 2).reshape(batch, frames, -1, self.width)
        hidden = tokens[:, 0]
        for index in range(1, frames):
            current = tokens[:, index]
            evidence = torch.cat(
                [
                    F.layer_norm(v.float(), (self.width,))
                    for v in (current, hidden, current - hidden)
                ],
                dim=-1,
            )
            gate = (self.gate_logit + self.content_gate(evidence).to(current.dtype)).sigmoid()
            hidden = gate * current + (1 - gate) * hidden
        hidden = hidden + self.vision.embeddings.position_embedding.weight.to(hidden.dtype)
        features = []
        for depth, layer in enumerate(self.vision.encoder.layers, 1):

            def forward(value, layer=layer):
                return layer(value, attention_mask=None)[0]

            if self.training and self.checkpoint_blocks and torch.is_grad_enabled():
                hidden = checkpoint(forward, hidden, use_reentrant=False)
            else:
                hidden = forward(hidden)
            if depth in self.depths:
                features.append(self.vision.post_layernorm(hidden))
        weights = self.depth_logits.softmax(0)
        fused = sum(weight * value for weight, value in zip(weights, features, strict=True))
        return self.semantic_projection(fused)


def build_semantic(config, *, initialize=False):
    """Architecture-only construction for export loading; never load DINO or a policy."""
    root = config["assets"]["starvla_root"]
    if root not in sys.path:
        sys.path.insert(0, root)
    from starVLA.model.modules.vlm.openpi_transformers.siglip.modeling_siglip import (
        SiglipVisionModel,
    )
    from transformers.models.auto import CONFIG_MAPPING

    arch = dict(config["student"]["vision_architecture"])
    arch.pop("projection_dim")
    cfg = CONFIG_MAPPING["paligemma"]().vision_config
    for key, value in arch.items():
        setattr(cfg, key, value)
    cfg.hidden_act = "gelu_pytorch_tanh"
    cfg.layer_norm_eps = 1e-6
    cfg._attn_implementation = "eager"
    tower = SiglipVisionModel(cfg)
    tower.vision_model.embeddings.patch_embedding = LocalConvStem(
        3,
        arch["hidden_size"],
        arch["patch_size"],
        conv_channels=config["student"]["stem_hidden"],
        seed=config["student"]["stem_seed"],
    )
    base = RawVisualEncoder(tower)
    if initialize:
        state = torch.load(Path(config["initial_encoder"]), map_location="cpu", weights_only=False)
        base.load_state_dict(state["encoder"], strict=True)
        del state
    return RawSemanticEncoder(
        base, output_width=config["semantic"]["width"], depths=config["semantic"]["depths"]
    )


def semantic_loss(student, teacher, *, gram_weight=2.0, global_weight=1.0):
    """Fixed semantic coordinates, dense relational structure, pooled scene semantics."""
    target = teacher.detach().float()
    s, t = F.normalize(student.float(), dim=-1), F.normalize(target, dim=-1)
    dense = (1 - (s * t).sum(-1)).mean()
    gram = F.mse_loss(s @ s.transpose(-1, -2), t @ t.transpose(-1, -2))
    global_loss = (1 - F.cosine_similarity(student.float().mean(1), target.mean(1), dim=-1)).mean()
    return dense + gram_weight * gram + global_weight * global_loss, {
        "dense_cosine_loss": dense,
        "gram_loss": gram,
        "global_cosine_loss": global_loss,
    }
