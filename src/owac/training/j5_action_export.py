"""Export a full J5/pi05 policy from two rank-local EMA shards, on CPU."""

import argparse
import json
from pathlib import Path

import torch

from owac.training.j5_pi05 import J5Policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    metadata = json.loads((args.checkpoint / "metadata.json").read_text())
    model = J5Policy(metadata["config"], initialize=False)
    kind = "model" if args.online else "ema"
    shards = [
        torch.load(args.checkpoint / f"{kind}-{rank}.pt", mmap=True, weights_only=False)
        for rank in range(2)
    ]
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(torch.cat([shard[name] for shard in shards], dim=0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": metadata["config"],
            "step": metadata["step"],
            "weights": kind,
        },
        args.output,
    )


if __name__ == "__main__":
    main()
