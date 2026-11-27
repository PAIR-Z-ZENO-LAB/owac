"""Train only a missing ISP; bounded loader and scalar-only training records."""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from owac.runtime.isp_policy import ISPPolicy
from owac.training.isp_benchmark_data import ISPData, collate_samples


def memory_usage():
    import psutil

    stat = dict(
        line.split() for line in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines()
    )
    return {
        "rss_gib": psutil.Process().memory_info().rss / 2**30,
        "cgroup_anon_gib": int(stat["anon"]) / 2**30,
        "cgroup_file_gib": int(stat["file"]) / 2**30,
        "cuda_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policy", choices=["corp", "pi05"], required=True)
    parser.add_argument("--frontend", choices=["rawild", "rawvla"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    recipe = cfg["training"]
    torch.set_num_threads(4)
    torch.manual_seed(recipe["seed"])
    random.seed(recipe["seed"])
    np.random.seed(recipe["seed"])
    model = ISPPolicy(cfg, args.policy, args.frontend)
    model.frontend.train()
    optimizer = torch.optim.AdamW(
        model.frontend.parameters(), lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-8
    )
    start = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.frontend.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start = state["step"]
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        del state
    dataset = ISPData(cfg, args.policy, args.frontend)
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch,
        sampler=range(start * recipe["batch"], len(dataset)),
        num_workers=2,
        prefetch_factor=1,
        persistent_workers=True,
        collate_fn=collate_samples,
    )
    iterator = iter(loader)
    accumulation = recipe["batch"] // args.micro_batch
    if accumulation * args.micro_batch != recipe["batch"]:
        raise ValueError("micro batch must divide the published batch of eight")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(
        json.dumps(
            {
                **cfg,
                "policy": args.policy,
                "frontend": args.frontend,
                "micro_batch": args.micro_batch,
            },
            indent=2,
        )
    )
    end = args.updates or recipe["updates"]
    with (args.output / "training.jsonl").open("a") as log:
        for step in range(start + 1, end + 1):
            started = time.monotonic()
            progress = (step - 100) / max(recipe["updates"] - 100, 1)
            rate = (
                1e-4 * step / 100
                if step < 100
                else 1e-6 + 0.5 * (1e-4 - 1e-6) * (1 + math.cos(math.pi * progress))
            )
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.zero_grad(set_to_none=True)
            values = {}
            temporal = 8 if args.frontend == "rawild" else 1
            for _ in range(accumulation):
                samples = next(iterator)
                temporal_batch = (4 if args.policy == "corp" else 2) if temporal == 8 else 1
                blocks = temporal // temporal_batch
                for block in range(blocks):
                    if temporal == 8:
                        window = [
                            {
                                **sample,
                                "raw": sample["raw"][:, index : index + 1],
                                "state": sample["state"][index : index + 1],
                                "actions": sample["actions"][index : index + 1],
                            }
                            for sample in samples
                            for index in range(block * temporal_batch, (block + 1) * temporal_batch)
                        ]
                    else:
                        window = samples
                    loss, parts = model.loss(window, 0)
                    (loss / (accumulation * blocks)).backward()
                    for key, value in {"total": loss, **parts}.items():
                        values[key] = values.get(key, 0.0) + float(value.detach()) / (
                            accumulation * blocks
                        )
                    del loss, parts, window
                del samples
            norm = torch.nn.utils.clip_grad_norm_(model.frontend.parameters(), 1.0)
            optimizer.step()
            torch.cuda.synchronize()
            if step == start + 1 or step % 10 == 0:
                record = {
                    "step": step,
                    "loss": values,
                    "gradient_norm": float(norm),
                    "lr": rate,
                    "seconds": time.monotonic() - started,
                    **memory_usage(),
                }
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
            if step == start + 1 or step % 1000 == 0 or step == end:
                payload = {
                    "step": step,
                    "policy": args.policy,
                    "frontend": args.frontend,
                    "model": model.frontend.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": cfg,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                    "python_rng": random.getstate(),
                    "numpy_rng": np.random.get_state(),
                }
                temporary = args.output / "latest.tmp"
                torch.save(payload, temporary)
                os.replace(temporary, args.output / "latest.pt")
                if step % 1000 == 0 or step == end:
                    torch.save(payload, args.output / f"step-{step}.pt")
                del payload
    (args.output / "complete.json").write_text(
        json.dumps(
            {"step": end, "native_exit": 0, "checkpoint": str(args.output / f"step-{end}.pt")}
        )
    )


if __name__ == "__main__":
    main()
