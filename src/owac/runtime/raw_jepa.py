"""Run the RAW JEPA visual pretraining stage."""

import argparse
import json
from pathlib import Path


def main():
    from owac.training.raw_jepa_train import train

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.resume:
        receipt = json.loads((args.resume.parent / "checkpoint.json").read_text())
        if not receipt["complete"]:
            raise RuntimeError("Full checkpoint write was incomplete; exact resume is unavailable")
    train(json.loads(args.config.read_text()), args.output, args.resume)


if __name__ == "__main__":
    main()
