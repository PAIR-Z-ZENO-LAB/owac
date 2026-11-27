"""Run independent full-resolution fruit continuations on the selected GPUs."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    spec = json.loads(args.sweep.read_text())
    variant = next(v for v in spec["variants"] if v["id"] == args.variant)
    cfg = json.loads((args.sweep.parent / spec["base_recipe"]).read_text())
    cfg["seed"] = args.seed if args.seed is not None else spec["seed"]
    cfg["continuation"] = spec["continuation"]
    cfg["training"].update(spec["training"], grip_motion_fraction=variant["motion"])
    if "grasp_evaluation" in spec:
        cfg["grasp_evaluation"] = spec["grasp_evaluation"]
        cfg["training"]["hand_pose_loss"] = variant.get("pose", False)
        cfg["model"]["spatial_coordinates"] = variant.get("xy", False)
        if variant.get("grounding"):
            cfg["model"]["spatial_goal_head"] = {"width": 128, "arms": 1}
            cfg["model"]["banana_grounding"] = True
            cfg["data"]["banana_labels"] = spec["banana_labels"]
            if variant.get("localizer"):
                cfg["assets"]["banana_head"] = variant["localizer"]
    cfg["model"]["freeze_raw_encoder"] = variant.get("freeze", False)
    if variant["stops"]:
        cfg["training"]["exposure_augmentation"] = {
            "probability": 0.5,
            "max_stops": variant["stops"],
        }
    cfg["evaluation"].update(spec["evaluation"])
    name = variant["id"] if args.seed is None else f"{variant['id']}-s{args.seed}"
    cfg["swanlab"].update(
        group=spec["group"],
        id=f"{spec['run_prefix']}-{name}",
        name=f"Fruit | {variant['label']} | seed {cfg['seed']} | +{cfg['training']['updates']}",
    )
    root = args.output / name
    root.mkdir(parents=True, exist_ok=True)
    recipe = root / "recipe.json"
    if recipe.exists() and json.loads(recipe.read_text()) != cfg:
        raise ValueError(f"A different recipe exists in {root}")
    recipe.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    status = root / "status.json"
    if status.exists() and json.loads(status.read_text())["status"] == "completed":
        return
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
    if (root / "latest.pt").exists():
        command += ["--resume", str(root / "latest.pt")]
    with (root / "train.log").open("a") as log:
        rc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
    (root / "train.exit").write_text(str(rc) + "\n")
    sys.exit(rc)


if __name__ == "__main__":
    main()
