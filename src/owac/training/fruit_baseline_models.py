"""Fruit policies using the released CoRP and StarVLA/OpenPI implementations."""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from owac.training.j5_lora import LoRALinear


def rawvla_offsets(recurrent_stride):
    """Six causal recurrent observations, each with six past or present burst frames."""
    past = np.arange(5, -1, -1)
    return -past[:, None] * recurrent_stride - past[None, :]


def masked_error(error, valid):
    return (error.mean(-1) * valid).sum() / valid.sum()


class DinoCoRP(nn.Module):
    """Run the public CoRP resampler and action generator on frozen DINOv3 patch tokens."""

    noise_dim = 8

    def __init__(self, cfg):
        super().__init__()
        from corp import load_policy

        self.cfg = cfg
        origin = load_policy(cfg["assets"]["corp"])
        rep = origin.representation_extractor
        self.views = rep.view_names
        self.resampler = rep.cross_attention_resampler
        self.state_encoder = rep.state_encoder
        self.generator = origin.action_generator
        if cfg.get("action_initialization") == "random":
            from corp.action_generator import DiTActionGenerator
            from corp.representation.cross_attention_resampler import CrossAttentionResampler
            from corp.representation.state_encoder import StateEncoder

            recipe = dict(
                origin.config,
                action_dim=cfg["model"]["action_dim"],
                action_horizon=cfg["model"]["action_horizon"],
            )
            self.resampler = CrossAttentionResampler(
                view_names=recipe["view_names"],
                hidden_dim=recipe["hidden_dim"],
                **recipe["representation_extractor"]["cross_attention_resampler"],
            )
            self.state_encoder = StateEncoder(cfg["model"]["state_dim"], recipe["hidden_dim"])
            self.generator = DiTActionGenerator(recipe)
        # A different visual basis, state semantics and embodiment need new interface layers.
        width = self.resampler.input_projection.out_features
        self.resampler.input_projection = nn.Linear(768, width)
        self.state_encoder.projection = nn.Linear(cfg["model"]["state_dim"], width)
        self.generator.action_dim = cfg["model"]["action_dim"]
        self.generator.action_projection = nn.Linear(self.generator.action_dim, width)
        self.generator.velocity_projection = nn.Linear(width, self.generator.action_dim)
        with torch.no_grad():
            language, pooled = rep.language_encoder([cfg["pi05"]["prompt"]])
        self.register_buffer("language", language)
        self.register_buffer("pooled_language", pooled)

    def prefix(self, batch):
        from corp.representation.extractor import Representation

        n = len(batch["state"])
        patches = dict(zip(self.views, batch["images"].unbind(1), strict=True))
        pooled = self.pooled_language.expand(n, -1)
        visual = self.resampler(patches, pooled)
        language = self.language.expand(n, -1, -1).to(visual.dtype)
        state = self.state_encoder(batch["state"])
        rep = Representation(torch.cat((visual, language, state), 1), pooled, visual.dtype)
        return self.generator.prepare_prefix(rep)

    def loss(self, batch):
        target = batch["actions"]
        prefix = self.prefix(batch)
        noise = torch.randn_like(target)
        tau = torch.rand(len(target), device=target.device)
        x = torch.lerp(noise, target, tau[:, None, None])
        v = self.generator(x, tau, prefix).float()
        loss = masked_error((v - (target - noise)).square(), batch["valid"])
        return loss, {"flow": loss.detach()}

    def sample(self, batch, noise):
        prefix = self.prefix(batch)
        steps = self.cfg["model"]["flow_steps"]
        action = noise
        for i in range(steps):
            tau = action.new_full((len(action),), i / steps)
            action = action + self.generator(action, tau, prefix).float() / steps
        return action


class FruitPi05(nn.Module):
    """A learned RAW frontend, public π0.5 base, and fruit-specific action-expert adaptation."""

    noise_dim = 32

    def __init__(self, cfg, method):
        super().__init__()
        from omegaconf import OmegaConf
        from starVLA.model.framework.VLM4A.PI05 import PI05
        from starVLA.model.modules.raw_frontend.registry import build_raw_frontend

        self.cfg, self.method = cfg, method
        pi = cfg["pi05"]
        origin = Path(cfg["assets"]["isp_run"])
        original = json.loads((origin / "config.json").read_text())
        frontend_cfg = (
            original["rawvla_frontend"]
            if method == "rawvla"
            else {"name": "rawild", "input_format": "rgb_raw"}
        )
        self.frontend = build_raw_frontend(frontend_cfg)
        ck = torch.load(
            origin / "training" / ("pi05-" + method) / "step-2000.pt",
            map_location="cpu",
            weights_only=False,
        )
        if ck["policy"] != "pi05" or ck["frontend"] != method:
            raise ValueError("Unexpected ISP checkpoint")
        self.frontend.load_state_dict(ck["model"], strict=True)
        fresh = cfg.get("action_initialization") == "random"
        if fresh:
            # Both learned frontends start with the same independently initialized action expert.
            torch.random.default_generator.manual_seed(cfg["seed"])
        self.policy = PI05(
            OmegaConf.create(
                {
                    "framework": {
                        "name": "PI05",
                        "precision": "bfloat16",
                        "action_horizon": cfg["model"]["action_horizon"],
                        "action_dim": 32,
                        "discrete_state_input": True,
                        "max_token_len": pi["max_token_len"],
                        "tokenizer": {
                            "model_path": cfg["assets"].get("tokenizer", original["tokenizer"]),
                            "pad_state_value": -2.0,
                        },
                        "raw_frontend": {"name": "none"},
                    }
                }
            )
        )
        weights = self.policy._normalize_state_dict(
            self.policy._load_checkpoint_state_dict(cfg["assets"]["pi05"])
        )
        if fresh:
            weights = {k: v for k, v in weights.items() if not k.startswith("action_head.")}
        result = self.policy.load_state_dict(weights, strict=False)
        unexpected = [
            k
            for k in result.missing_keys
            if not self.policy._is_expected_openpi_missing_key(k)
            and not (fresh and k.startswith("action_head."))
        ]
        if unexpected or result.unexpected_keys:
            raise ValueError(f"Unexpected OpenPI parameters {unexpected} {result.unexpected_keys}")
        del weights
        self.policy.image_keys = self.policy.image_keys[:2]  # Only these cameras have observations.
        head = self.policy.action_head
        if fresh:
            self.policy.vlm.requires_grad_(False)
            head.float()
            # The continuous action expert never uses the vocabulary output projection.
            head.gemma_expert.lm_head.requires_grad_(False).to(torch.bfloat16)
            self.policy.eval()
            return
        self.policy.requires_grad_(False)
        for layer in (head.action_in_proj, head.action_out_proj):
            layer.float().requires_grad_(True)
        rank = pi["lora_rank"]
        for layer in head.gemma_expert.model.layers:
            for name in ("q_proj", "v_proj"):
                setattr(
                    layer.self_attn,
                    name,
                    LoRALinear(getattr(layer.self_attn, name), rank, rank * 2),
                )
        self.policy.eval()

    def train(self, mode=True):
        super().train(mode)
        self.policy.eval()  # Released vision/language inference behavior throughout fine-tuning.
        return self

    def observation(self, batch):
        images = batch["images"].flatten(0, 1).float()
        rgb = self.frontend(images, input_format="rgb_raw")
        rgb = rgb.reshape(len(batch["state"]), 2, *rgb.shape[1:])
        states = batch["state_q"].detach().cpu().numpy()
        examples = [
            {"image": list(views), "lang": self.cfg["pi05"]["prompt"], "state": q}
            for views, q in zip(rgb, states, strict=True)
        ]
        return self.policy._build_observation_from_batch(
            self.policy._prepare_examples(examples, include_actions=False)
        )

    def loss(self, batch):
        observation = self.observation(batch)
        dim = self.cfg["model"]["action_dim"]
        target = F.pad(batch["actions"], (0, self.noise_dim - dim))
        values = self.policy.forward_from_observation(observation, target, return_debug=True)
        err = (values["v_t"][..., :dim].float() - values["u_t"][..., :dim]).square()
        flow = masked_error(err, batch["valid"])
        components = {"flow": flow.detach()}
        if self.method == "rawvla":
            auxiliary = self.frontend.last_aux_losses
            exposure, chroma = auxiliary["exposure_prior_loss"], auxiliary["chroma_prior_loss"]
            # Rescale the author's joint objective to unit action-loss weight for matched action LR.
            loss = flow + (0.01 * exposure + 0.001 * chroma) / 0.15
            components.update(exposure=exposure.detach(), chroma=chroma.detach())
            return loss, components
        return flow, components

    def sample(self, batch, noise):
        out = self.policy.sample_actions(
            self.observation(batch), noise=noise, num_steps=self.cfg["pi05"]["flow_steps"]
        )
        return out[..., : self.cfg["model"]["action_dim"]].float()


def weights(model):
    return {k: v for k, v in model.named_parameters() if v.requires_grad}


def export(model):
    state = {k: v.detach().cpu() for k, v in weights(model).items()}
    # Includes running frontend statistics and the frozen representation of the single instruction.
    for name, tensor in model.named_buffers():
        if not name.startswith("policy."):
            state[name] = tensor.detach().cpu()
    return state


def load_export(model, state):
    expected = set(weights(model)) | {
        n for n, _ in model.named_buffers() if not n.startswith("policy.")
    }
    if expected != set(state):
        raise ValueError(f"Checkpoint keys differ {expected.symmetric_difference(state.keys())}")
    model.load_state_dict(state, strict=False)


def make_model(cfg, method):
    if method == "dino-corp":
        return DinoCoRP(cfg)
    if method in ("rawvla", "rawild"):
        return FruitPi05(cfg, method)
    raise ValueError(method)
