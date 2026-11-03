"""Compare retained foundation weights to official pi05 and quantify effective LoRA edits."""

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

from owac.training.j5_lora import LoRALinear, build_policy, copy_trainable


def official_name(name):
    match = re.match(r"paligemma_with_expert.joint_layers.(\d+).streams.([01]).(.*)", name)
    if match:
        index, stream, tail = match.groups()
        prefix = "paligemma.model.language_model" if stream == "0" else "gemma_expert.model"
        return f"paligemma_with_expert.{prefix}.layers.{index}.{tail.replace('.base.', '.')}"
    if name.endswith("embed_tokens.weight"):
        return (
            "paligemma_with_expert.paligemma.lm_head.weight"
            if ".paligemma." in name
            else "paligemma_with_expert.gemma_expert.lm_head.weight"
        )
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    payload = torch.load(args.checkpoint, mmap=True, map_location="cpu", weights_only=False)
    model = build_policy(payload["config"])
    copy_trainable(model, payload["ema"])
    rows = []
    with safe_open(payload["config"]["assets"]["pi05"], framework="pt", device="cpu") as original:
        for name, p in model.named_parameters():
            if name.startswith("raw_") or name.endswith((".a.weight", ".b.weight")):
                continue
            key = official_name(name)
            base = original.get_tensor(key).to(p.dtype)
            equal = torch.equal(p, base)
            rows.append(
                dict(
                    name=name,
                    official_key=key,
                    parameters=p.numel(),
                    dtype=str(p.dtype),
                    exact_after_dtype_cast=equal,
                    max_abs=0.0
                    if equal
                    else float((p.detach().float() - base.float()).abs().max()),
                )
            )
    edits = []
    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        a, b = module.a.weight.detach().double(), module.b.weight.detach().double()
        norm = float(((a @ a.T) * (b.T @ b)).sum().clamp_min(0).sqrt()) * module.scale
        base_norm = float(module.base.weight.detach().float().norm())
        edits.append(
            dict(
                name=name,
                stream="language" if ".streams.0." in name else "expert",
                effective_lora_norm=norm,
                base_norm=base_norm,
                relative_lora_norm=norm / base_norm,
            )
        )
    result = dict(
        checkpoint=str(args.checkpoint),
        foundation_parameters=sum(x["parameters"] for x in rows),
        foundation_tensors=len(rows),
        all_foundation_exact=all(x["exact_after_dtype_cast"] for x in rows),
        foundation=rows,
        lora=edits,
    )
    args.output.write_text(json.dumps(result, indent=2))
    print(
        json.dumps({k: v for k, v in result.items() if k not in ["foundation", "lora"]}), flush=True
    )
    for stream in ["language", "expert"]:
        values = [x for x in edits if x["stream"] == stream]
        print(
            stream,
            "relative effective update",
            (
                sum(x["effective_lora_norm"] ** 2 for x in values)
                / sum(x["base_norm"] ** 2 for x in values)
            )
            ** 0.5,
            flush=True,
        )


if __name__ == "__main__":
    main()
