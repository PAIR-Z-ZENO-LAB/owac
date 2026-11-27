"""Continue the deployed fruit regressor with a matched pickup or placement sampler."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variant", choices=("pickup-control", "basket-balanced", "carry-release"), required=True
    )
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    if args.variant == "carry-release":
        spec.update(spec["carry_release"])
    checkpoint = Path(spec["checkpoint"])
    prior = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    cfg = prior["config"]
    with checkpoint.open("rb") as f:
        cfg["continuation"] = dict(
            checkpoint=str(checkpoint),
            step=prior["step"],
            sha256=hashlib.file_digest(f, "sha256").hexdigest(),
            preserve_statistics=True,
        )
    cfg["seed"] = spec["seed"]
    cfg["training"].update(
        {k: spec[k] for k in ("updates", "eval_every", "warmup", "learning_rate")}
    )
    cfg["training"].update(
        save_every=spec["updates"], balance_pick_and_place=args.variant == "basket-balanced"
    )
    if cfg["training"]["balance_pick_and_place"]:
        cfg["training"]["approach_fraction"] = 0
    if args.variant == "carry-release":
        cfg["training"].update(approach_fraction=0, placement_only=True)
    cfg["evaluation"].update(placement=True, metric="placement/selection_mm")
    cfg["swanlab"].update(
        id=f"fruit-place20-{args.variant}",
        name=f"Fruit | basket | {args.variant}",
        group="real-fruit-placement-20261120",
    )
    root = args.output / args.variant
    root.mkdir(parents=True, exist_ok=True)
    recipe = root / "recipe.json"
    if recipe.exists() and json.loads(recipe.read_text()) != cfg:
        raise ValueError(f"Existing recipe differs in {root}")
    recipe.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=1",
        "--module",
        "owac.training.teleop_policy_train",
        "--config",
        str(recipe),
        "--output",
        str(root),
    ]
    if (root / "latest.pt").is_file():
        command.extend(["--resume", str(root / "latest.pt")])
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
