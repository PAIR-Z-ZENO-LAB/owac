"""Expand a compact action checkpoint to train RAW vision, preserving existing Adam state."""

import argparse
import json
from pathlib import Path

import torch

from owac.training.j5_action_train import disk_save
from owac.training.j5_lora import (
    build_policy,
    configure_lora,
    copy_trainable,
    parameter_groups,
    trainable,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    payload = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    cfg = json.loads(args.config.read_text())
    model = build_policy(payload["config"])
    copy_trainable(model, payload["model"])
    old = torch.optim.AdamW(parameter_groups(model, payload["config"]["training"]))
    old.load_state_dict(payload["optimizer"])
    configure_lora(model, cfg)
    optimizer = torch.optim.AdamW(
        parameter_groups(model, cfg["training"]),
        lr=cfg["training"]["learning_rate"],
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-10,
        foreach=True,
    )
    old_groups = {g["name"]: g for g in old.param_groups}
    for group in optimizer.param_groups:
        if group["name"] in old_groups:
            group.update({k: v for k, v in old_groups[group["name"]].items() if k != "params"})
    optimizer.state.update(old.state)
    parameters = trainable(model)
    cfg["selection"] = payload["config"]["selection"]
    result = dict(payload)
    result.update(
        config=cfg,
        model={n: p.detach() for n, p in parameters.items()},
        ema={
            n: payload["ema"][n] if n in payload["ema"] else p.detach().clone()
            for n, p in parameters.items()
        },
        optimizer=optimizer.state_dict(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    disk_save(result, args.output)
    receipt = dict(
        source=str(args.checkpoint),
        step=payload["step"],
        inherited_optimizer_states=len(old.state),
        vision_parameters=sum(p.numel() for p in model.raw_encoder.parameters()),
        trainable_parameters=sum(p.numel() for p in parameters.values()),
        inherited_weight_max_abs=max(
            float((parameters[n].detach() - v).abs().max()) for n, v in payload["model"].items()
        ),
        encoder_initial_optimizer_states=sum(
            p in optimizer.state for p in model.raw_encoder.parameters()
        ),
    )
    args.output.with_suffix(".json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
