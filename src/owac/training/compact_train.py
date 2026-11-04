"""Resumable joint RAW CoRP training with masked 7D flow matching and EMA."""

import argparse
import copy
import json
import math
import os
import random
import signal
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from owac.frameworks.raw_compact.policy import CompactPolicy
from owac.training.compact_data import CompactFrames
from owac.training.j5_action_metrics import action_metrics, action_sums


def atomic_save(payload, path):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def move(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def validate(model, dataset, rank, world, device):
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=32,
        sampler=range(rank, len(dataset), world),
        num_workers=2,
        generator=torch.Generator().manual_seed(dataset.cfg["seed"] + rank),
        pin_memory=True,
    )
    totals = torch.zeros(1 + len(dataset.lighting_names), 8, dtype=torch.float64, device=device)
    stats = {
        "q01": dataset.cfg["statistics"]["action_q01"],
        "q99": dataset.cfg["statistics"]["action_q99"],
    }
    for batch in loader:
        batch = move(batch, device)
        noise = []
        for sid, endpoint in zip(
            batch["source_index"].cpu().tolist(), batch["endpoint"].cpu().tolist(), strict=True
        ):
            generator = torch.Generator(device=device).manual_seed(
                20261104 + sid * 100003 + endpoint
            )
            noise.append(
                torch.randn(batch["actions"].shape[1:], generator=generator, device=device)
            )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model.sample(batch, torch.stack(noise))
        totals[0] += action_sums(
            prediction,
            batch["actions"],
            batch["action_valid"],
            stats,
            dataset.cfg["closed_loop"]["execute_horizon"],
        )
        for index in range(len(dataset.lighting_names)):
            selected = batch["lighting_id"] == index
            totals[index + 1] += action_sums(
                prediction[selected],
                batch["actions"][selected],
                batch["action_valid"][selected],
                stats,
                dataset.cfg["closed_loop"]["execute_horizon"],
            )
    dist.all_reduce(totals)
    result = {k: float(v) for k, v in action_metrics(totals[0]).items()}
    for index, name in enumerate(dataset.lighting_names):
        result.update(
            {
                k.replace("validation/", f"validation/{name}/"): float(v)
                for k, v in action_metrics(totals[index + 1]).items()
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-step", type=int)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    train = cfg["training"]
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.set_num_threads(8)
    dist.init_process_group("nccl", device_id=device)
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"] + rank)
    np.random.seed(cfg["seed"] + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = CompactPolicy(cfg, initialize=args.resume is None).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train["learning_rate"],
        weight_decay=train["weight_decay"],
        betas=(0.9, 0.95),
        fused=True,
    )
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    start = 0
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False, mmap=True)
        model.load_state_dict(payload["model"])
        ema.load_state_dict(payload["ema"])
        optimizer.load_state_dict(payload["optimizer"])
        start = payload["step"]
        rng = payload["rng"][rank]
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state(rng["cuda"], device)
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        del payload
    else:
        torch.manual_seed(cfg["seed"] + rank)
    wrapped = DDP(model, device_ids=[device.index], broadcast_buffers=False)
    dataset = CompactFrames(cfg)
    validation = CompactFrames(cfg, split="validation")
    loader = DataLoader(
        dataset,
        batch_size=train["micro_batch"],
        sampler=range(start * train["batch"] + rank, len(dataset), world),
        num_workers=train["workers"],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=1,
        generator=torch.Generator().manual_seed(cfg["seed"] + rank),
    )
    iterator = iter(loader)
    accumulation = train["batch"] // (world * train["micro_batch"])
    assert accumulation * world * train["micro_batch"] == train["batch"]
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    swanlab = None
    if rank == 0:
        (root / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
        (root / "model.json").write_text(
            json.dumps(
                {
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "trainable_parameters": sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    ),
                }
            )
        )
        if not args.offline:
            import swanlab

            swanlab.login(api_key=Path("/root/.config/owac/swanlab-api-key").read_text().strip())
            swanlab.merge_settings(swanlab.Settings(probe=swanlab.Settings.Probe(monitor=False)))
            swanlab.init(
                project="OWAC",
                workspace="Jeong",
                id=cfg["swanlab"]["id"],
                resume="allow",
                experiment_name=cfg["swanlab"]["name"],
                group=cfg["swanlab"]["group"],
                job_type="raw-proprio-language-flow",
                config=cfg,
                logdir=str(root / "swanlab"),
            )
            for section, label in [
                ("objective", "01 训练目标"),
                ("validation", "02 动作验证"),
                ("optimizer", "03 优化器"),
                ("runtime", "04 运行资源"),
            ]:
                swanlab.define_metric(f"{section}/*", section_name=label)
    stop_requested = [False]
    signal.signal(signal.SIGUSR1, lambda *_: stop_requested.__setitem__(0, True))

    def report(step, metrics):
        if rank == 0:
            with (root / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": step, "metrics": metrics}) + "\n")
            print(json.dumps({"step": step, **metrics}), flush=True)
            if swanlab:
                swanlab.log(metrics, step=step)

    def save(step, retain=False):
        state = {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device),
            "python": random.getstate(),
            "numpy": np.random.get_state(),
        }
        rngs = [None] * world
        dist.all_gather_object(rngs, state)
        if rank == 0:
            payload = {
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "rng": rngs,
                "config": cfg,
            }
            atomic_save(payload, root / "latest.pt")
            if retain:
                atomic_save(payload, root / f"step-{step:06d}.pt")
            atomic_save(
                {"model": ema.state_dict(), "step": step, "config": cfg}, root / "policy.pt"
            )
        dist.barrier()

    stop_step = args.stop_step or train["updates"]
    model.train()
    running = torch.zeros(4, device=device)
    running_updates = 0
    clock = time.monotonic()
    for step in range(start + 1, stop_step + 1):
        progress = max(0, step - train["warmup"]) / (train["updates"] - train["warmup"])
        lr = (
            train["learning_rate"]
            * min(step / train["warmup"], 1)
            * (0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2)
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        for micro in range(accumulation):
            batch = move(next(iterator), device)
            context = wrapped.no_sync() if micro + 1 < accumulation else nullcontext()
            with context:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, flow, task = wrapped(batch)
                    scaled = loss / accumulation
                scaled.backward()
            running[:3] += torch.stack((loss.detach(), flow, task)) / accumulation
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            torch._foreach_lerp_(
                list(ema.parameters()), list(model.parameters()), 1 - train["ema_decay"]
            )
        running[3] += grad.detach()
        running_updates += 1
        should_stop = torch.tensor(int(stop_requested[0]), device=device)
        dist.all_reduce(should_stop, op=dist.ReduceOp.MAX)
        if step % train["log_every"] == 0 or step == start + 1:
            dist.all_reduce(running)
            count = running_updates
            values = (running / (world * count)).tolist()
            elapsed = time.monotonic() - clock
            report(
                step,
                dict(
                    zip(
                        [
                            "objective/loss",
                            "objective/flow7",
                            "objective/task",
                            "optimizer/grad_norm",
                        ],
                        values,
                        strict=True,
                    )
                )
                | {
                    "optimizer/lr": lr,
                    "runtime/update_seconds": elapsed / count,
                    "runtime/samples_per_second": count * train["batch"] / elapsed,
                    "runtime/gpu_peak_allocated_gib": torch.cuda.max_memory_allocated(device)
                    / 2**30,
                },
            )
            running.zero_()
            running_updates = 0
            clock = time.monotonic()
            if rank == 0:
                (root / "status.json").write_text(
                    json.dumps({"step": step, "status": "running", "target_step": stop_step})
                )
        ending = step == stop_step or bool(should_stop)
        if step % train["eval_every"] == 0 or ending:
            report(step, validate(ema, validation, rank, world, device))
            model.train()
            clock = time.monotonic()
        if step % train["save_every"] == 0 or step == start + 1 or ending:
            save(step, retain=ending)
        if ending:
            if rank == 0:
                (root / "status.json").write_text(
                    json.dumps(
                        {"step": step, "status": "paused" if bool(should_stop) else "completed"}
                    )
                )
            break
    if swanlab:
        swanlab.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
