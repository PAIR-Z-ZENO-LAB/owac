"""J5 visual replacement of official openpi PI0Pytorch, with shardable joint layers.

JointLayer follows openpi 15a9616 models_pytorch/gemma_pytorch.py; SDPA replaces
materialized eager attention without changing the prefix/suffix attention mask.
The action objective and timestep sampler remain inherited from PI0Pytorch.
"""

import math
from types import MethodType, SimpleNamespace

import torch
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers.models.gemma import modeling_gemma

from owac.training.raw_ssl import RawEncoder


class JointLayer(nn.Module):
    def __init__(self, language, expert):
        super().__init__()
        self.streams = nn.ModuleList([language, expert])

    def forward(self, streams, conds, mask, cos, sin):
        queries, keys, values, gates = [], [], [], []
        for layer, hidden, cond in zip(self.streams, streams, conds, strict=True):
            normalized, gate = layer.input_layernorm(hidden, cond=cond)
            shape = (*normalized.shape[:-1], -1, layer.self_attn.head_dim)
            queries.append(layer.self_attn.q_proj(normalized).view(shape).transpose(1, 2))
            keys.append(layer.self_attn.k_proj(normalized).view(shape).transpose(1, 2))
            values.append(layer.self_attn.v_proj(normalized).view(shape).transpose(1, 2))
            gates.append(gate)
        q, k = modeling_gemma.apply_rotary_pos_emb(
            torch.cat(queries, 2), torch.cat(keys, 2), cos, sin, unsqueeze_dim=1
        )
        v = torch.cat(values, 2)
        # Gemma MQA: expand keys/values over query heads (views, not copied repeats).
        k, v = k.expand(-1, q.shape[1], -1, -1), v.expand(-1, q.shape[1], -1, -1)
        attended = (
            F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask.to(q.dtype), scale=self.streams[0].self_attn.scaling
            )
            .transpose(1, 2)
            .reshape(q.shape[0], q.shape[2], -1)
        )
        result, offset = [], 0
        for layer, hidden, gate, cond in zip(self.streams, streams, gates, conds, strict=True):
            end = offset + hidden.shape[1]
            update = layer.self_attn.o_proj(attended[:, offset:end])
            hidden = modeling_gemma._gated_residual(hidden, update, gate)
            normalized, gate = layer.post_attention_layernorm(hidden, cond=cond)
            result.append(modeling_gemma._gated_residual(hidden, layer.mlp(normalized), gate))
            offset = end
        return result


def joint_forward(
    self,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    adarms_cond=None,
):
    if inputs_embeds[0] is None or inputs_embeds[1] is None:
        return self.original_forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            adarms_cond=adarms_cond,
        )
    conds = [None, None] if adarms_cond is None else adarms_cond
    dummy = inputs_embeds[0].new_empty((inputs_embeds[0].shape[0], position_ids.shape[1], 256))
    cos, sin = self.paligemma.model.language_model.rotary_emb(dummy, position_ids)
    streams = inputs_embeds
    for block in self.joint_layers:
        if self.training and torch.is_grad_enabled():
            streams = checkpoint(
                block,
                streams,
                conds,
                attention_mask,
                cos,
                sin,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            streams = block(streams, conds, attention_mask, cos, sin)
    models = [self.paligemma.language_model, self.gemma_expert.model]
    return [
        model.norm(x, cond=cond)[0] for model, x, cond in zip(models, streams, conds, strict=True)
    ], None


class J5Policy(PI0Pytorch):
    def __init__(self, cfg, *, initialize=True):
        super().__init__(
            Pi0Config(
                pi05=True,
                action_horizon=10,
                discrete_state_input=bool(cfg.get("proprioception")),
                dtype="float32",
                pytorch_compile_mode=None,
            )
        )
        if initialize:
            from safetensors.torch import load_model

            load_model(self, cfg["assets"]["pi05"], strict=True)
        j5 = torch.load(cfg["assets"]["j5"], map_location="cpu", weights_only=False, mmap=True)
        self.raw_encoder = RawEncoder(j5["config"]["model"])
        self.raw_encoder.load_state_dict(j5["encoder"])
        self.raw_projector = nn.Linear(j5["config"]["model"]["width"], 2048)
        self.raw_encoder.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        pg = self.paligemma_with_expert
        # No SigLIP/RGB image processing or language-generation logits in this policy.
        pg.paligemma.model.vision_tower = None
        pg.paligemma.model.multi_modal_projector = None
        pg.paligemma.lm_head = None
        pg.gemma_expert.lm_head = None
        language, expert = pg.paligemma.language_model, pg.gemma_expert.model
        left, right = list(language.layers), list(expert.layers)
        del language.layers, expert.layers
        # The official cached inference path still accesses the same layer objects.
        object.__setattr__(language, "layers", left)
        object.__setattr__(expert, "layers", right)
        pg.joint_layers = nn.ModuleList(
            [JointLayer(a, b) for a, b in zip(left, right, strict=True)]
        )
        object.__setattr__(pg, "original_forward", pg.forward)
        pg.forward = MethodType(joint_forward, pg)
        self.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "freeze_raw_encoder", False):
            self.raw_encoder.eval()
        return self

    def _preprocess_observation(self, observation, *, train=True):
        # Dataset handles RAW-safe geometric augmentation; no RGB range conversion.
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        batch = images[0].shape[0]
        _, tokens = self.raw_encoder(torch.cat(images[:2], dim=0))
        projected = self.raw_projector(tokens)
        first, second = projected.split(batch)
        language = self.paligemma_with_expert.embed_language_tokens(lang_tokens) * math.sqrt(2048)
        # The right-wrist slot is entirely masked in LIBERO. Eliding it preserves
        # all valid-token positions/masks and avoids 256 useless transformer queries.
        prefix = torch.cat((first, second, language), dim=1)
        masks = torch.cat(
            (*[mask[:, None].expand(-1, 256) for mask in img_masks[:2]], lang_masks), dim=1
        )
        return prefix, masks, torch.zeros_like(masks)

    def forward(self, batch, noise=None, time=None):
        return super().forward(observation(batch), batch["actions"], noise=noise, time=time)

    @torch.no_grad()
    def sample_sharded(self, batch, noise, num_steps=10):
        """Official Euler velocity integration using joint forwards instead of KV cache.

        This path works with FSDP joint-layer hooks during validation. Deployment
        uses the inherited cached sample_actions with the exported unsharded policy.
        """
        obs = observation(batch)
        images, masks, ids, id_mask, state = self._preprocess_observation(obs, train=False)
        prefix, pad, att = self.embed_prefix(images, masks, ids, id_mask)
        x = noise
        for index in range(num_steps):
            t = torch.full((len(x),), 1 - index / num_steps, device=x.device)
            suffix, suffix_pad, suffix_att, cond = self.embed_suffix(state, x, t)
            full_pad = torch.cat((pad, suffix_pad), 1)
            full_att = torch.cat((att, suffix_att), 1)
            mask = self._prepare_attention_masks_4d(make_att_2d_masks(full_pad, full_att))
            (_, result), _ = self.paligemma_with_expert.forward(
                attention_mask=mask,
                position_ids=full_pad.long().cumsum(1) - 1,
                inputs_embeds=[prefix, suffix],
                adarms_cond=[None, cond],
                use_cache=False,
            )
            velocity = self.action_out_proj(result[:, -10:].float()).float()
            x = x - velocity / num_steps
        return x


def observation(batch):
    raw = batch["raw"]
    yes = torch.ones(len(raw), dtype=torch.bool, device=raw.device)
    return SimpleNamespace(
        images={
            "base_0_rgb": raw[:, 0],
            "left_wrist_0_rgb": raw[:, 1],
            "right_wrist_0_rgb": torch.zeros_like(raw[:, 0]),
        },
        image_masks={"base_0_rgb": yes, "left_wrist_0_rgb": yes, "right_wrist_0_rgb": ~yes},
        tokenized_prompt=batch["tokens"],
        tokenized_prompt_mask=batch["token_mask"],
        token_ar_mask=None,
        token_loss_mask=None,
        state=batch.get("state", torch.zeros((len(raw), 32), device=raw.device)),
    )
