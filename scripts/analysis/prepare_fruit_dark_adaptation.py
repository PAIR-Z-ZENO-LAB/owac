"""Make fruit action recipes with a common actor and an independently trained RAW encoder."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch


def recipes(spec, root):
    checkpoint = torch.load(
        spec["parent_policy"], map_location="cpu", mmap=True, weights_only=False
    )
    base = checkpoint["config"]
    for name, variant in spec["variants"].items():
        cfg = copy.deepcopy(base)
        cfg["seed"] = spec["seed"]
        cfg["continuation"] = {
            "checkpoint": spec["parent_policy"],
            "sha256": spec["parent_sha256"],
            "step": checkpoint["step"],
            "new_visual_step": 12288,
            "preserve_statistics": True,
        }
        cfg["assets"].pop("banana_head", None)
        cfg["assets"].update(
            visual_encoder=spec["visual_encoder"], visual_sha256=spec["visual_sha256"]
        )
        cfg["data"].pop("banana_labels", None)
        cfg["data"].update(
            inventory=spec["inventory"],
            dataset_revision=spec["dataset_revision"],
            feature_cache=str(root / "features"),
            action_subsets=variant["action_subsets"],
        )
        cfg["model"]["freeze_raw_encoder"] = variant["frozen_steps"] == spec["updates"]
        training = cfg["training"]
        training.pop("exposure_augmentation", None)
        for key in (
            "updates",
            "batch",
            "eval_every",
            "learning_rate",
            "visual_lr_multiplier",
            "warmup",
            "ema_decay",
        ):
            training[key] = spec[key]
        training.update(
            frozen_steps=variant["frozen_steps"],
            save_every=spec["updates"],
            banana_supervision=False,
        )
        cfg["evaluation"] = {
            "validation_episodes": spec["validation_episodes"],
            "metric": "grasp/fruit03_closing_position_mm",
            "grasp_by_subset": True,
        }
        cfg["swanlab"].update(
            id=f"fruit-dark20-12288-{name}",
            name=f"Fruit | RAW 12288 | {name}",
            group="real-fruit-dark-action-20261120",
        )
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "recipe.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))


def paired_recipes(spec, root):
    phase = spec["paired_continuation"]
    path = root / phase["parent"]
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    with path.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    for name in ("native-cont", "paired-head"):
        cfg = copy.deepcopy(checkpoint["config"])
        cfg["continuation"] = dict(
            checkpoint=str(path), sha256=digest, step=checkpoint["step"], preserve_statistics=True
        )
        tc = cfg["training"]
        tc.update(frozen_steps=0, save_every=phase["updates"])
        for key in ("updates", "learning_rate", "warmup"):
            tc[key] = phase[key]
        if name == "paired-head":
            tc["paired_low_light"] = {
                "fit": str(root / phase["fit"]),
                "weight": phase["weight"],
                "source_subsets": phase["source_subsets"],
            }
        cfg["swanlab"].update(id=f"fruit-dark20-12288-{name}", name=f"Fruit | RAW 12288 | {name}")
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "recipe.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))


def action_midpoint(spec, root):
    """Check retention of the deployed controller with a fixed action-weight average."""
    path = root / "paired-head" / "policy-best.pt"
    adapted = torch.load(path, mmap=True, map_location="cpu", weights_only=False)
    original = torch.load(spec["parent_policy"], mmap=True, map_location="cpu", weights_only=False)
    for key in adapted["model"]:
        value = adapted["model"][key]
        if not key.startswith("raw_encoder.") and torch.is_floating_point(value):
            adapted["model"][key] = torch.lerp(original["model"][key], value, 0.5)
    adapted["config"]["action_weight_average"] = {
        "adapted": str(path),
        "deployed": spec["parent_policy"],
        "adapted_fraction": 0.5,
        "vision": "unchanged from adapted checkpoint",
    }
    out = root / "action-midpoint"
    out.mkdir(exist_ok=True)
    torch.save(adapted, out / "policy.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--paired-stage", action="store_true")
    stage.add_argument("--action-midpoint", action="store_true")
    args = parser.parse_args()
    build = recipes
    if args.paired_stage:
        build = paired_recipes
    if args.action_midpoint:
        build = action_midpoint
    build(json.loads(args.spec.read_text()), args.output.resolve())
