"""Check official-backend preservation using identical RGB prefix tokens and RAW sampling paths."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from owac.evaluation.j5_regression_probe import compact_policy
from owac.runtime.j5_policy_server import ActionPolicy
from owac.training.j5_lora import LoRALinear


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official-precision", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    raw = compact_policy(args.checkpoint, "cuda:0")
    cfg = torch.load(args.checkpoint, mmap=True, map_location="cpu", weights_only=False)["config"]
    rgb = ActionPolicy(cfg, cfg["assets"]["pi05"], True, "cuda:0")
    if args.official_precision:
        for name, parameter in raw.model.named_parameters():
            if name.startswith("raw_") or name.endswith((".a.weight", ".b.weight")):
                continue
            if not name.startswith("paligemma_with_expert.") or any(
                key in name for key in ("input_layernorm", "post_attention_layernorm", "model.norm")
            ):
                parameter.data = parameter.data.float()
    raw_embed = raw.model.embed_prefix
    rgb_embed = rgb.model.embed_prefix
    captured = []

    def capture(*a, **kw):
        result = rgb_embed(*a, **kw)
        captured[:] = [result]
        return result

    rgb.model.embed_prefix = capture
    adapters = [m for m in raw.model.modules() if isinstance(m, LoRALinear)]
    paths = [
        Path(p)
        for p in json.loads(args.index.read_text())["paths"]
        if int(Path(p).stem) in [0, 40, 80]
    ]
    rows = []
    for p in paths:
        with np.load(p) as z:
            request = dict(
                raw=z["raw"], rgb=z["rgb"], prompt=str(z["prompt"].item()), noise=z["noise"]
            )
        ref = rgb.infer(request)["normalized_actions"]
        raw.model.embed_prefix = lambda *a, **kw: captured[0]
        for m in adapters:
            m.scale = 0.0
        zero = raw.infer(request)["normalized_actions"]
        for m in adapters:
            m.scale = 1.0
        adapted = raw.infer(request)["normalized_actions"]
        raw.model.embed_prefix = raw_embed
        cached = raw.infer(request)["normalized_actions"]
        prompt = request["prompt"].strip().replace("_", " ").replace("\n", " ")
        ids = (raw.tokenizer.encode(prompt, add_bos=True) + raw.tokenizer.encode("\n"))[:200]
        batch = dict(
            raw=torch.tensor(request["raw"], device="cuda:0").permute(0, 3, 1, 2).float()[None]
            / 255,
            tokens=torch.tensor([ids], device="cuda:0"),
            token_mask=torch.ones(1, len(ids), device="cuda:0", dtype=torch.bool),
        )
        noise = torch.tensor(request["noise"], device="cuda:0")[None]
        pg = raw.model.paligemma_with_expert
        forward = pg.forward
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            joint = raw.model.sample_sharded(batch, noise)[0, :, :7].float().cpu().numpy()
            pg.forward = pg.original_forward
            eager = raw.model.sample_sharded(batch, noise)[0, :, :7].float().cpu().numpy()
            pg.forward = forward

        def diff(a, b):
            return dict(
                rmse=float(np.sqrt(np.mean((a[:5] - b[:5]) ** 2))),
                max_abs=float(np.abs(a - b).max()),
                gripper_flip=float(np.mean((a[:5, 6] > 0) != (b[:5, 6] > 0))),
            )

        rows.append(
            dict(
                path=str(p),
                official_vs_zero_lora=diff(ref, zero),
                official_vs_adapted_lora=diff(ref, adapted),
                cached_vs_joint=diff(cached, joint),
                eager_vs_joint=diff(eager, joint),
            )
        )
        print(len(rows), json.dumps(rows[-1]), flush=True)
    args.output.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
