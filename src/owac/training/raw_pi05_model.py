"""Reuse the verified J5/official-LIBERO backend and its existing adapter weights."""

import copy

import torch

from owac.training.j5_lora import configure_lora
from owac.training.j5_pi05 import J5Policy


def adaptation_state(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if key.startswith(("raw_encoder.", "raw_projector."))
        or ".a.weight" in key
        or ".b.weight" in key
    }


def build_model(cfg, checkpoint=None):
    torch.manual_seed(cfg["seed"])
    model = J5Policy(cfg, initialize=True)
    architecture = copy.deepcopy(cfg)
    architecture["adaptation"]["ranks"] = [16, 8]
    configure_lora(model, architecture)
    payload = torch.load(
        checkpoint or cfg["warm_start"], map_location="cpu", mmap=True, weights_only=False
    )
    state = payload.get("ema", payload["model"])
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise ValueError(f"Unexpected inherited adapter parameters: {result.unexpected_keys}")
    # The historical file contains every bridge and LoRA weight, but not its
    # frozen J5 or official base. Those were strictly restored above.
    required = {
        name
        for name in model.state_dict()
        if name.startswith("raw_projector.") or ".a.weight" in name or ".b.weight" in name
    }
    if not required.issubset(state):
        raise ValueError(f"Incomplete inherited adapter: {sorted(required - state.keys())}")
    del payload, state
    configure_lora(model, cfg)
    model.raw_encoder.backbone.embeddings.mask_token.requires_grad_(False)
    return model
