"""Package the original pickup and the trained transport action head around their shared encoder."""

import argparse
from pathlib import Path

import torch

from owac.frameworks.raw_compact.fruit_placement import FruitPlacementPolicy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("pickup", "carry", "gate", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--aperture", type=Path)
    args = p.parse_args()
    parent = torch.load(args.pickup, mmap=True, map_location="cpu", weights_only=False)
    carry = torch.load(args.carry, mmap=True, map_location="cpu", weights_only=False)
    cfg = parent["config"]
    for name in ("observation", "statistics", "model", "task_token_ids", "task_token_mask"):
        if cfg[name] != carry["config"][name]:
            raise ValueError(f"Expert configuration differs: {name}")
    visual = [k for k in parent["model"] if k.startswith("raw_encoder.")]
    if not visual or any(not torch.equal(parent["model"][k], carry["model"][k]) for k in visual):
        raise ValueError("The experts do not share the same frozen RAW encoder")
    cfg["model"]["kind"] = "fruit_placement_regression"
    if args.aperture:
        aperture = torch.load(args.aperture, map_location="cpu", weights_only=True)
        cfg["model"]["release_landmarks"] = len(aperture["releases"])
    cfg["placement_components"] = dict(
        pickup=str(args.pickup),
        carry=str(args.carry),
        gate=str(args.gate),
        shared_visual_tensors=len(visual),
    )
    model = FruitPlacementPolicy(cfg)
    model.pickup.load_state_dict(parent["model"], strict=True)
    model.carry.load_state_dict(
        {k: v for k, v in carry["model"].items() if k not in visual}, strict=True
    )
    model.gate.load_state_dict(
        torch.load(args.gate, map_location="cpu", weights_only=True), strict=True
    )
    if args.aperture:
        cfg["placement_components"]["aperture"] = str(args.aperture)
        model.aperture.load_state_dict(aperture, strict=True)
    torch.save(dict(model=model.state_dict(), config=cfg, step=carry["step"]), args.output)


if __name__ == "__main__":
    main()
