"""Isolate inherited RAW encoder training stochasticity with fixed policy/noise."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from owac.evaluation.j5_regression_probe import compact_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    policy = compact_policy(args.root / "normal/initial-003000.pt", "cuda:0")
    paths = [
        Path(p)
        for p in json.loads((args.root / "weight-audit/probe-index.json").read_text())["paths"]
        if int(Path(p).stem) in [0, 40, 80]
    ]
    requests = []
    for p in paths:
        with np.load(p) as z:
            requests.append(dict(raw=z["raw"], prompt=str(z["prompt"].item()), noise=z["noise"]))
    rows = []
    predictions = []
    for mode in ["eval", "rope_only", "drop_path_only", "raw_train"]:
        encoder = policy.model.raw_encoder
        encoder.eval()
        if mode == "rope_only":
            encoder.backbone.rope_embeddings.train()
        if mode == "drop_path_only":
            for module in encoder.modules():
                if type(module).__name__ == "DINOv3ViTDropPath":
                    module.train()
        if mode == "raw_train":
            encoder.train()
        results = []
        for i, request in enumerate(requests):
            repeated = []
            for repeat in range(1 if mode == "eval" else 8):
                torch.manual_seed(91000 + i * 8 + repeat)
                repeated.append(policy.infer(request)["normalized_actions"])
            results.append(np.stack(repeated))
        values = np.stack(results)
        predictions.append(values)
        if mode != "eval":
            reference = predictions[0]
            rows.append(
                dict(
                    mode=mode,
                    observations=len(paths),
                    repeats=8,
                    action_rmse_from_eval=float(
                        np.sqrt(np.mean((values[:, :, :5] - reference[:, :, :5]) ** 2))
                    ),
                    gripper_sign_change=float(
                        np.mean((values[:, :, :5, 6] > 0) != (reference[:, :, :5, 6] > 0))
                    ),
                    repeated_action_std_rms=float(np.sqrt(np.mean(values[:, :, :5].var(axis=1)))),
                )
            )
        print(mode, rows[-1] if rows else len(paths), flush=True)
        np.savez_compressed(args.root / f"weight-audit/train-mode-{mode}.npz", actions=values)
    (args.root / "weight-audit/train-mode-probe.json").write_text(
        json.dumps(dict(paths=[str(p) for p in paths], results=rows), indent=2)
    )


if __name__ == "__main__":
    main()
