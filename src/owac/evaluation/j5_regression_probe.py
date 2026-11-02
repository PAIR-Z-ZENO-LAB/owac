"""Compare checkpoint and RGB-reference actions on identical replay observations."""

import argparse
import json
from pathlib import Path
from types import MethodType

import numpy as np
import sentencepiece
import torch

from owac.runtime.j5_policy_server import ActionPolicy
from owac.training.j5_lora import build_policy, copy_trainable


def compact_policy(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    cfg = payload["config"]
    policy = object.__new__(ActionPolicy)
    policy.device, policy.baseline = torch.device(device), False
    policy.model = build_policy(cfg)
    copy_trainable(policy.model, payload["ema"])
    policy.model.to(device).eval()
    policy.tokenizer = sentencepiece.SentencePieceProcessor(model_file=cfg["assets"]["tokenizer"])
    policy.stats = json.loads(Path(cfg["assets"]["statistics"]).read_text())["norm_stats"][
        "actions"
    ]
    policy.infer = MethodType(ActionPolicy.infer, policy)
    return policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = args.root
    out = root / "regression-audit"
    cfg = json.loads((root / "production.json").read_text())
    policies = {}
    for step, device in [(3000, "cuda:0"), (6000, "cuda:1")]:
        policies[str(step)] = compact_policy(root / f"normal/step-{step:06d}.pt", device)
        print("loaded", step, flush=True)
    policies["rgb"] = ActionPolicy(cfg, cfg["simulation"]["rgb_checkpoint"], True, "cuda:1")
    print("loaded rgb", flush=True)
    rows, arrays = [], []
    for path in sorted((out / "inputs").glob("*/*/*.npz")):
        with np.load(path) as z:
            source, case = path.parent.parent.name, path.parent.name
            actions = []
            for name, policy in policies.items():
                result = policy.infer(
                    {
                        "rgb" if name == "rgb" else "raw": z["rgb" if name == "rgb" else "raw"],
                        "prompt": str(z["prompt"].item()),
                        "noise": z["noise"],
                    }
                )
                actions.append(result["normalized_actions"])
                if name == source:
                    replay_delta = float(np.abs(result["actions"] - z["recorded_plan"]).max())
            actions = np.stack(actions)
            arrays.append(actions)
            rows.append(
                dict(
                    source=source,
                    case=case,
                    step=int(z["step"]),
                    replay_action_max_abs=replay_delta,
                    checkpoint_delta_rmse=float(
                        np.sqrt(np.mean((actions[0, :5] - actions[1, :5]) ** 2))
                    ),
                    rgb_rmse_3000=float(np.sqrt(np.mean((actions[0, :5] - actions[2, :5]) ** 2))),
                    rgb_rmse_6000=float(np.sqrt(np.mean((actions[1, :5] - actions[2, :5]) ** 2))),
                    gripper_flip_fraction=float(
                        np.mean((actions[0, :5, 6] > 0) != (actions[1, :5, 6] > 0))
                    ),
                )
            )
        if len(rows) % 20 == 0:
            print("observations", len(rows), flush=True)
    (out / "same-observation.json").write_text(json.dumps(rows, indent=2))
    np.savez_compressed(out / "same-observation-actions.npz", actions=np.stack(arrays))
    print("completed", len(rows), flush=True)


if __name__ == "__main__":
    main()
