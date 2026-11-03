"""Measure checkpoint drift and isolate module updates on identical saved observations."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from owac.evaluation.j5_regression_probe import compact_policy
from owac.training.j5_lora import copy_trainable


def group(name):
    if name.startswith("raw_encoder."):
        return "vision"
    if name.startswith("raw_projector."):
        return "bridge"
    return "language" if ".streams.0." in name else "expert"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    old = torch.load(args.before, map_location="cpu", mmap=True, weights_only=False)["ema"]
    new = torch.load(args.after, map_location="cpu", mmap=True, weights_only=False)["ema"]
    totals = {
        k: dict(old_squared=0.0, delta_squared=0.0, parameters=0, changed=0)
        for k in ["vision", "bridge", "language", "expert"]
    }
    layers = []
    for n, value in new.items():
        delta = value.float() - old[n].float()
        a, b = float(old[n].float().square().sum()), float(delta.square().sum())
        row = dict(
            name=n,
            group=group(n),
            old_norm=a**0.5,
            delta_norm=b**0.5,
            relative_delta=(b / a) ** 0.5 if a else None,
            max_abs=float(delta.abs().max()),
        )
        layers.append(row)
        t = totals[group(n)]
        t["old_squared"] += a
        t["delta_squared"] += b
        t["parameters"] += value.numel()
        t["changed"] += int(b > 0)
    effective = []
    for n in new:
        if not n.endswith(".a.weight"):
            continue
        bn = n.removesuffix("a.weight") + "b.weight"

        def norm2(a, b):
            return float(((a @ a.T) * (b.T @ b)).sum().clamp_min(0))

        a0, a1, b0, b1 = old[n].double(), new[n].double(), old[bn].double(), new[bn].double()
        d = norm2(torch.cat([a1, a0], 0), torch.cat([b1, -b0], 1))
        base = norm2(a0, b0)
        effective.append(
            dict(
                name=n.removesuffix(".a.weight"),
                group=group(n),
                old_norm=base**0.5,
                delta_norm=d**0.5,
                relative_delta=(d / base) ** 0.5 if base else None,
            )
        )
    for t in totals.values():
        t["relative_delta"] = (t["delta_squared"] / t["old_squared"]) ** 0.5
    (args.output / "weight-drift.json").write_text(
        json.dumps(dict(groups=totals, tensors=layers, lora_effective=effective), indent=2)
    )
    print(json.dumps(totals), flush=True)
    policy = compact_policy(args.after, args.device)
    paths = [
        p
        for p in sorted(args.inputs.glob("*/*/*.npz"))
        if int(p.stem) % 20 == 0 or ("libero_object" in str(p) and int(p.stem) <= 80)
    ]
    modes = {
        "before": set(),
        "after": {"vision", "bridge", "language", "expert"},
        "vision": {"vision"},
        "bridge": {"bridge"},
        "language": {"language"},
        "expert": {"expert"},
        "without_expert": {"vision", "bridge", "language"},
    }
    arrays = []
    for mode, groups in modes.items():
        copy_trainable(policy.model, {n: new[n] if group(n) in groups else old[n] for n in new})
        actions = []
        for p in paths:
            with np.load(p) as z:
                actions.append(
                    policy.infer(
                        dict(raw=z["raw"], prompt=str(z["prompt"].item()), noise=z["noise"])
                    )["normalized_actions"]
                )
        arrays.append(np.stack(actions))
        np.savez_compressed(args.output / f"actions-{mode}.npz", actions=arrays[-1])
        print(mode, len(actions), flush=True)
    (args.output / "probe-index.json").write_text(
        json.dumps(dict(modes=list(modes), paths=[str(p) for p in paths]), indent=2)
    )
    np.savez_compressed(args.output / "module-actions.npz", actions=np.stack(arrays))


if __name__ == "__main__":
    main()
