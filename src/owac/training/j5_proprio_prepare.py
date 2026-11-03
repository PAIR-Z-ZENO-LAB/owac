"""Fit train-only joint/gripper quantiles and preserve a full continuation state."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from owac.raw.all_bench_distillation import load_inventory
from owac.training.j5_action_train import disk_save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    sources, _ = load_inventory(cfg["data"]["inventory"])
    sources = [
        s
        for s in sources
        if s.domain == "libero" and s.split == "train" and s.lighting in cfg["data"]["lighting"]
    ]
    values = []
    for source in sources:
        with np.load(source.path, allow_pickle=False) as archive:
            values.append(archive["states"][:, cfg["proprioception"]["state_columns"]])
    state = np.concatenate(values)
    low, high = np.quantile(state, [0.01, 0.99], axis=0)
    cfg["proprioception"]["statistics"] = {"q01": low.tolist(), "q99": high.tolist()}
    cfg["proprioception"]["statistics_sources"] = len(sources)
    cfg["proprioception"]["statistics_frames"] = len(state)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(cfg, indent=2))
    payload = torch.load(
        cfg["continuation"]["checkpoint"], map_location="cpu", weights_only=False, mmap=True
    )
    # State input has no new weights. Preserve model/Adam/RNG; only reset selection
    # because validation under the new observation contract is a new baseline.
    cfg["selection"] = {
        "best": None,
        "bad_evals": 0,
        "lr_factor": 1.0,
        "metric": cfg["training"]["selection_metric"],
    }
    payload["config"] = cfg
    disk_save(payload, args.output / "initial-003000.pt")
    print(
        json.dumps(
            {
                "sources": len(sources),
                "frames": len(state),
                "step": payload["step"],
                "optimizer_states": len(payload["optimizer"]["state"]),
            }
        )
    )


if __name__ == "__main__":
    main()
