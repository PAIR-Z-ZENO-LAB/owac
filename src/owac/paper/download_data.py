"""Fetch only the public NPZ files listed in the paper split."""

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from owac.paper.prepare import ROOT

DATASET = "ToferFish/RawVLA-Bench"
REVISION = "8586c594fff9ce38580e29745f074e82e96e74f7"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--domain", choices=("libero", "robotwin2", "both"), default="both")
    parser.add_argument("--split", type=Path, default=ROOT / "data/manifests/paper-rawvla.jsonl")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    groups = [json.loads(line) for line in args.split.read_text().splitlines()]
    paths = [
        f"data/{g['domain']}/raw/{g['relpath']}"
        for g in groups
        if args.domain in ("both", g["domain"])
    ]
    if len(paths) != len(set(paths)) or not paths:
        parser.error("Expected unique trajectories in the published split")
    snapshot_download(
        DATASET,
        repo_type="dataset",
        revision=REVISION,
        allow_patterns=paths,
        local_dir=args.dataset,
        max_workers=args.workers,
    )
    for rel in paths:
        if not (args.dataset / rel).is_file():
            raise FileNotFoundError(args.dataset / rel)
    print(f"Ready: {len(paths)} RAW trajectory files under {args.dataset.resolve()}")


if __name__ == "__main__":
    main()
