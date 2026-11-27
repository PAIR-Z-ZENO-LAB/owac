"""Bind the native-Bayer encoder and action recipes to locally owned recordings and weights."""

import argparse
import hashlib
import json
from pathlib import Path

from owac.paper.prepare import ROOT, write


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("vision", "action"))
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--encoder", required=True, type=Path)
    parser.add_argument("--anchor", type=Path, help="J3 encoder for the vision relation objective")
    parser.add_argument("--language", type=Path, help="local T5 directory for action training")
    parser.add_argument("--instruction", help="single task instruction for these recordings")
    args = parser.parse_args()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    groups = json.loads(args.inventory.read_text())["groups"]
    if {g["domain"] for g in groups} != {"real_teleop"} or {g["split"] for g in groups} != {
        "train",
        "validation",
    }:
        parser.error("Expected a native-Bayer inventory with a training and a validation split")
    folder = ROOT / "configs/experiments"
    if args.stage == "vision":
        if not args.anchor:
            parser.error("Vision adaptation requires --anchor")
        cfg = json.loads((folder / "real_fruit_raw_ssl_8192.json").read_text())
        cfg["training"]["initialize_encoder_from"] = str(args.encoder.resolve(strict=True))
        cfg["relations"]["anchor"] = str(args.anchor.resolve(strict=True))
    else:
        if not (args.language and args.instruction):
            parser.error("Action training requires --language and --instruction")
        cfg = json.loads((folder / "real_fruit_action_regression.json").read_text())
        cfg["task_names"] = [args.instruction]
        cfg["assets"].update(
            language_encoder=str(args.language.resolve(strict=True)),
            visual_encoder=str(args.encoder.resolve(strict=True)),
            visual_sha256=digest(args.encoder),
        )
        cfg["data"].update(
            dataset_revision=digest(args.inventory), feature_cache=str(work / "features")
        )
        cfg["evaluation"]["validation_episodes"] = sorted(
            g["group_id"] for g in groups if g["split"] == "validation"
        )
    cfg["data"]["inventory"] = str(args.inventory.resolve())
    write(work / (args.stage + ".json"), cfg)


if __name__ == "__main__":
    main()
