"""Capture actual cached-policy tokens and action attention during recorded replay."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers.models.gemma import modeling_gemma

from owac.runtime.j5_policy_server import ActionPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    cfg = json.loads(args.config.read_text())
    policy = ActionPolicy(cfg, args.checkpoint, args.baseline, "cuda:0")
    model = policy.model
    layers = [0, 8, 17]
    expert_ids = {
        id(model.paligemma_with_expert.gemma_expert.model.layers[i].self_attn): i for i in layers
    }
    captured = {}
    attention = {}

    def raw_hook(module, inputs, output):
        captured["raw_tokens"] = output[1].float().cpu().numpy()

    def projected_hook(module, inputs, output):
        captured["projected_tokens"] = output.float().cpu().numpy()

    def contextual_hook(module, inputs, output):
        captured["contextual_tokens"] = output[0][0, :512].reshape(2, 256, -1).float().cpu().numpy()

    if args.baseline:
        vision_outputs, projection_outputs = [], []

        def vision_hook(module, inputs, output):
            vision_outputs.append(output.last_hidden_state.float().cpu().numpy())

        def projection_hook(module, inputs, output):
            projection_outputs.append(output.float().cpu().numpy())

        pg = model.paligemma_with_expert.paligemma.model
        hooks = [
            pg.vision_tower.register_forward_hook(vision_hook),
            pg.multi_modal_projector.register_forward_hook(projection_hook),
        ]
    else:
        hooks = [
            model.raw_encoder.register_forward_hook(raw_hook),
            model.raw_projector.register_forward_hook(projected_hook),
        ]
    hooks.append(
        model.paligemma_with_expert.paligemma.language_model.norm.register_forward_hook(
            contextual_hook
        )
    )
    original = modeling_gemma.eager_attention_forward

    def traced_attention(module, *a, **kw):
        output, weights = original(module, *a, **kw)
        if id(module) in expert_ids:
            # Actual softmax probabilities, averaged over the five executed action positions.
            attention[expert_ids[id(module)]].append(
                weights[0, :, :5].float().mean(1).cpu().numpy()
            )
        return output, weights

    modeling_gemma.eager_attention_forward = traced_attention
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = []
    try:
        for path in sorted(args.inputs.glob("*.npz")):
            data = np.load(path)
            captured.clear()
            if args.baseline:
                vision_outputs.clear()
                projection_outputs.clear()
            attention = {i: [] for i in layers}
            result = policy.infer(
                {
                    ("rgb" if args.baseline else "raw"): data["raw"],
                    "prompt": str(data["prompt"]),
                    "noise": data["noise"],
                }
            )
            if args.baseline:
                captured["raw_tokens"] = np.concatenate(vision_outputs[:2])
                captured["projected_tokens"] = np.concatenate(projection_outputs[:2])
            maps = np.stack([np.stack(attention[i]) for i in layers])
            # Dimensions: selected layer, denoising iteration, attention head, key token.
            np.savez_compressed(
                args.output / path.name,
                **captured,
                attention=maps,
                actions=result["actions"],
                layers=layers,
            )
            averaged = maps.mean((0, 1, 2))
            delta = np.abs(result["actions"] - data["recorded_plan"])
            row = {
                "case": path.stem,
                "action_max_abs_difference": float(delta.max()),
                "action_mean_abs_difference": float(delta.mean()),
                "attention_mass": {
                    "external_camera": float(averaged[:256].sum()),
                    "wrist_camera": float(averaged[256:512].sum()),
                    "text": float(averaged[768 if args.baseline else 512 : -10].sum()),
                    "action": float(averaged[-10:].sum()),
                },
            }
            for name, tokens in captured.items():
                x = torch.from_numpy(tokens[0])
                x = torch.nn.functional.normalize(x, dim=-1)
                sim = x @ x.T
                row[name + "/mean_off_diagonal_cosine"] = float(
                    (sim.sum() - sim.diag().sum()) / (256 * 255)
                )
            metrics.append(row)
            print(json.dumps(row), flush=True)
    finally:
        modeling_gemma.eager_attention_forward = original
        for hook in hooks:
            hook.remove()
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
