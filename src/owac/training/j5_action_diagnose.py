"""Read-only, fixed-input diagnosis of a paused J5 LoRA policy; no optimizer steps."""

import argparse
import json
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import torch

from owac.raw.all_bench_distillation import load_inventory
from owac.training.j5_action_data import RawActionFrames, collate_actions
from owac.training.j5_action_train import move
from owac.training.j5_lora import LoRALinear, build_policy, copy_trainable, trainable
from owac.training.j5_pi05 import observation


def group(name):
    if name.startswith("raw_encoder"):
        return "raw_encoder"
    if name.startswith("raw_projector"):
        return "raw_projector"
    if ".streams.0." in name:
        return "language_lora"
    if ".streams.1." in name:
        return "expert_lora"
    return "action_interfaces"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda:0")
    payload = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    cfg = payload["config"]
    model = build_policy(cfg).to(device).eval()
    sources, _ = load_inventory(cfg["data"]["inventory"])
    dataset = RawActionFrames(
        [s for s in sources if s.domain == "libero" and s.split == "validation"],
        cfg,
        validation=True,
    )
    lights = ["Normal", "Low", "ExtremeLow", "Over", "ExtremeOver"]
    selected = [
        i
        for light in lights
        for i in [j for j, s in enumerate(dataset.sources) if s.lighting == light][:4]
    ]
    result = {
        "step": payload["step"],
        "checkpoint": str(args.checkpoint),
        "validation_count": len(dataset),
        "sample_indices": selected,
        "ema_order_matches_model": list(payload["ema"]) == list(trainable(model)),
        "optimizer_steps": sorted({int(s["step"]) for s in payload["optimizer"]["state"].values()}),
        "cases": {},
    }

    def batch_at(i):
        batch = move(collate_actions([dataset[i]]), device)
        torch.manual_seed(70000 + i)
        noise = torch.randn_like(batch["actions"])
        time = model.sample_time(1, device)
        return batch, noise, time

    def evaluate(label, *, fp32=False, indices=None):
        rows = []
        with torch.no_grad():
            for i in range(len(dataset)) if indices is None else indices:
                batch, noise, time = batch_at(i)
                context = nullcontext() if fp32 else torch.autocast("cuda", dtype=torch.bfloat16)
                with context:
                    loss = model(batch, noise=noise, time=time)
                    row = {
                        "index": i,
                        "lighting": batch["lighting"][0],
                        "time": time.item(),
                        "flow32": loss.mean().item(),
                        "flow7": loss[..., :7].mean().item(),
                        "padding25": loss[..., 7:].mean().item(),
                    }
                    if i in selected:
                        pred = model.sample_actions(
                            device, observation(batch), noise=noise, num_steps=10
                        )
                        row["sample_mae7"] = (
                            (pred[..., :7] - batch["actions"][..., :7]).abs().mean().item()
                        )
                rows.append(row)
        result["cases"][label] = rows
        summary = {k: sum(r[k] for r in rows) / len(rows) for k in ("flow32", "flow7", "padding25")}
        print(json.dumps({"case": label, **summary}), flush=True)
        (args.output / "results.json").write_text(json.dumps(result, indent=2))

    for kind in ("model", "ema"):
        copy_trainable(model, payload[kind])
        evaluate(kind)
    copy_trainable(model, payload["model"])
    modules = [m for m in model.modules() if isinstance(m, LoRALinear)]
    for m in modules:
        m.scale *= 0.5
    evaluate("online_lora_half")
    for m in modules:
        m.scale *= 2
    base = torch.load(cfg["adaptation"]["base"], map_location="cpu", mmap=True, weights_only=False)
    for prefix in ("raw_encoder", "raw_projector"):
        copy_trainable(model, payload["model"])
        with torch.no_grad():
            for name, p in trainable(model).items():
                if name.startswith(prefix):
                    p.copy_(base["model"][name])
        evaluate("online_reset_" + prefix)
    del base
    copy_trainable(model, payload["model"])

    # Separate the weighted 7D and padded-25D contributions to the official loss.
    parameters = trainable(model)
    energy = defaultdict(lambda: torch.zeros(3, device=device))
    for i in selected:
        batch, noise, time = batch_at(i)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(batch, noise=noise, time=time)
        real = torch.autograd.grad(
            loss[..., :7].mean() * 7 / 32,
            tuple(parameters.values()),
            retain_graph=True,
            allow_unused=True,
        )
        padded = torch.autograd.grad(
            loss[..., 7:].mean() * 25 / 32, tuple(parameters.values()), allow_unused=True
        )
        for name, a, b in zip(parameters, real, padded, strict=True):
            if a is not None and b is not None:
                energy[group(name)] += torch.stack(
                    (a.square().sum(), b.square().sum(), (a * b).sum())
                ).detach()
        del loss, real, padded
    result["gradient_energy_weighted_loss"] = {
        k: (v / len(selected)).cpu().tolist() for k, v in energy.items()
    }
    result["gradient_energy_columns"] = [
        "real7_squared_norm",
        "padding25_squared_norm",
        "dot_product",
    ]
    print(json.dumps({"gradient_energy": result["gradient_energy_weighted_loss"]}), flush=True)
    model.float()
    evaluate("online_fp32", fp32=True, indices=selected)
    (args.output / "results.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
