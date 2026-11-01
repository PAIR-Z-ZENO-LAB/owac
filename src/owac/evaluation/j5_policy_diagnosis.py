"""Fixed-noise input interventions on saved native initial observations."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from owac.runtime.j5_policy_server import ActionPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", choices=["raw", "rgb"], required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cfg = json.loads(args.config.read_text())
    if args.port:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy

        policy = WebsocketClientPolicy(host="127.0.0.1", port=args.port)
    else:
        policy = ActionPolicy(cfg, args.checkpoint, args.role == "rgb", args.device)
    prompts = json.loads(args.prompts.read_text())
    rows = [
        json.loads(line) for line in (args.episodes / "episodes.jsonl").read_text().splitlines()
    ]
    inputs, recorded = [], []
    for row in rows:
        folder = args.episodes / f"{row['suite']}-00-Normal-{row['initial_index']:02d}"
        with np.load(folder / "initial.npz") as z:
            inputs.append({key: z[key].copy() for key in z.files})
        with np.load(folder / "actions.npz") as z:
            recorded.append(z["plans"][0].copy())
    results, arrays = [], []
    for index, (row, inp) in enumerate(zip(rows, inputs, strict=True)):
        other = next(i for i, item in enumerate(rows) if item["suite"] != row["suite"])
        request = {args.role: inp["inputs"], "prompt": prompts[row["suite"]], "noise": inp["noise"]}
        original = policy.infer(request)
        swapped_prompt = policy.infer(dict(request, prompt=prompts[rows[other]["suite"]]))
        swapped_image = policy.infer({**request, args.role: inputs[other]["inputs"]})
        norms = np.stack(
            [item["normalized_actions"] for item in [original, swapped_prompt, swapped_image]]
        )
        arrays.append(norms)
        entry = {
            "suite": row["suite"],
            "initial_index": row["initial_index"],
            "replayed_action_max_abs": float(np.abs(original["actions"] - recorded[index]).max()),
            "prompt_swap_rmse": float(np.sqrt(np.mean((norms[0, :5] - norms[1, :5]) ** 2))),
            "image_swap_rmse": float(np.sqrt(np.mean((norms[0, :5] - norms[2, :5]) ** 2))),
            "original_actions_first5": original["actions"][:5].tolist(),
        }
        if args.role == "raw" and row["initial_index"] == 5:
            cleaned = request["prompt"].strip().replace("_", " ").replace("\n", " ")
            ids = (policy.tokenizer.encode(cleaned, add_bos=True) + policy.tokenizer.encode("\n"))[
                :200
            ]
            tokens = torch.tensor([ids], device=policy.device)
            batch = {
                "raw": torch.tensor(inp["inputs"], device=policy.device)
                .permute(0, 3, 1, 2)
                .float()[None]
                / 255,
                "tokens": tokens,
                "token_mask": torch.ones_like(tokens, dtype=torch.bool),
            }
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                sampled = policy.model.sample_sharded(
                    batch, torch.tensor(inp["noise"], device=policy.device)[None], num_steps=10
                )
            delta = sampled[0, :, :7].float().cpu().numpy() - norms[0]
            entry["joint_vs_cached_rmse"] = float(np.sqrt(np.mean(delta**2)))
            entry["joint_vs_cached_max_abs"] = float(np.abs(delta).max())
        results.append(entry)
        print(json.dumps(entry), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2))
    np.savez(args.output / "actions.npz", normalized=np.stack(arrays))


if __name__ == "__main__":
    main()
