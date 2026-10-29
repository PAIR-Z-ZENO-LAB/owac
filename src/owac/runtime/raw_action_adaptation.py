"""Train R1 for the configured optimizer updates, or resume its complete state."""

import argparse
import json
from pathlib import Path


def main():
    from owac.training.raw_action_adaptation import train

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    train(json.loads(args.config.read_text()), output=args.output, resume=args.resume)


if __name__ == "__main__":
    main()
