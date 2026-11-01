"""LoRA adaptation and compact resumable checkpoints for the J5 action policy."""

import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.scale = alpha / rank
        self.a = nn.Linear(base.in_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        return self.base(x) + self.b(self.a(x)) * self.scale


class ResidualProjector(nn.Module):
    """Keep the learned bridge and add a zero-initialized nonlinear correction."""

    def __init__(self, base, hidden=1024):
        super().__init__()
        self.base = base
        self.residual = nn.Sequential(
            nn.LayerNorm(base.in_features),
            nn.Linear(base.in_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, base.out_features),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, x):
        return self.base(x) + self.residual(x)


def configure_lora(model, cfg):
    if cfg["adaptation"].get("residual_projector") and not isinstance(
        model.raw_projector, ResidualProjector
    ):
        model.raw_projector = ResidualProjector(model.raw_projector)
    model.requires_grad_(False)
    # Legacy runs train the dense front end; staged runs name only the bridge.
    for name, module in model.named_children():
        if name in cfg["adaptation"].get(
            "dense_modules", [n for n, _ in model.named_children() if n != "paligemma_with_expert"]
        ):
            module.requires_grad_(True)
    for block in model.paligemma_with_expert.joint_layers:
        for stream, layer in enumerate(block.streams):
            rank = cfg["adaptation"]["ranks"][stream]
            if rank == 0:
                continue
            for parent, names in (
                (layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")),
                (layer.mlp, ("gate_proj", "up_proj", "down_proj")),
            ):
                for name in names:
                    if name not in cfg["adaptation"].get("targets_list", names):
                        continue
                    existing = getattr(parent, name)
                    if isinstance(existing, LoRALinear):
                        existing.a.requires_grad_(True)
                        existing.b.requires_grad_(True)
                        continue
                    setattr(
                        parent,
                        name,
                        LoRALinear(
                            getattr(parent, name), rank, rank * cfg["adaptation"]["alpha_over_rank"]
                        ),
                    )
    model.freeze_raw_encoder = not any(p.requires_grad for p in model.raw_encoder.parameters())
    if model.freeze_raw_encoder:
        model.raw_encoder.eval()
    # Frozen backend needs no FP32 master/Adam copy. Keep accepted RAW weights FP32.
    for name, p in model.named_parameters():
        if not p.requires_grad and not name.startswith("raw_encoder."):
            p.data = p.data.to(torch.bfloat16)
    return model


def build_policy(cfg):
    from owac.training.j5_pi05 import J5Policy

    if cfg["adaptation"].get("fresh_official", False):
        torch.manual_seed(cfg["seed"])
        return configure_lora(J5Policy(cfg, initialize=True), cfg)
    model = J5Policy(cfg, initialize=False)
    path = Path(cfg["adaptation"]["base"])
    payload = torch.load(path, mmap=True, weights_only=False, map_location="cpu")
    if cfg["adaptation"].get("base_has_adapters"):
        configure_lora(model, payload["config"])
    model.load_state_dict(payload["model"])
    del payload
    with path.open("rb") as stream:
        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return configure_lora(model, cfg)


def trainable(model):
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


def parameter_groups(model, training):
    grouped = {name: [] for name in training["group_learning_rates"]}
    for name, parameter in trainable(model).items():
        if name.startswith("raw_projector."):
            group = "bridge"
        elif ".streams.0." in name:
            group = "language"
        elif ".q_proj." in name or ".k_proj." in name:
            group = "expert_qk"
        else:
            group = "expert_content"
        grouped[group].append(parameter)
    return [
        {
            "name": name,
            "params": values,
            "lr_scale": training["group_learning_rates"][name] / training["learning_rate"],
        }
        for name, values in grouped.items()
    ]


@torch.no_grad()
def copy_trainable(model, state):
    for name, p in trainable(model).items():
        p.copy_(state[name])


def save(model, optimizer, ema, root, step, cfg, rank):
    from owac.training.j5_action_train import disk_save

    rng = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(),
    }
    states = [None] * dist.get_world_size() if rank == 0 else None
    dist.gather_object(rng, states, dst=0)
    if rank == 0:
        disk_save(
            {
                "step": step,
                "config": cfg,
                "model": {k: p.detach().cpu() for k, p in trainable(model).items()},
                "ema": {k: p.cpu() for k, p in ema.items()},
                "optimizer": optimizer.state_dict(),
                "rng": states,
            },
            root / "saving.pt",
        )
        (root / "saving.pt").replace(root / "latest.pt")
    dist.barrier()


def load(model, optimizer, path, rank):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    copy_trainable(model, payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    device = next(model.parameters()).device
    ema = {k: p.to(device) for k, p in payload["ema"].items()}
    rng = payload["rng"][rank]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    torch.cuda.set_rng_state(rng["cuda"])
    return payload["step"], ema
