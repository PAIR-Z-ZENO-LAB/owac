"""Full pi05 fine-tuning: official flow objective/settings, streamed RAW, FSDP2."""

import argparse
import gc
import json
import os
import random
import shutil
import signal
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard, register_fsdp_forward_method
from torch.distributed.tensor import DTensor
from torch.utils.data import DataLoader, Sampler

from owac.raw.all_bench_distillation import load_inventory
from owac.training.j5_action_data import RawActionFrames, collate_actions
from owac.training.j5_pi05 import J5Policy


def local(value):
    return value.to_local() if isinstance(value, DTensor) else value


@contextmanager
def ema_weights(model, ema):
    online = {name: local(p).detach().cpu() for name, p in model.named_parameters()}
    with torch.no_grad():
        for name, p in model.named_parameters():
            local(p).copy_(ema[name])
    try:
        yield
    finally:
        # Root weights stay gathered after no-grad forwards (no backward hook).
        model.reshard()
        with torch.no_grad():
            for name, p in model.named_parameters():
                local(p).copy_(online[name])


def disk_save(payload, path):
    # Flush/drop our checkpoint pages instead of keeping tens of GB of write cache.
    with path.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def save_rank(model, optimizer, ema, temporary, rank):
    state = {k: local(v).detach().cpu() for k, v in model.state_dict().items()}
    disk_save(state, temporary / f"model-{rank}.pt")
    del state
    state = [
        {
            k: local(v).detach().cpu() if torch.is_tensor(v) else v
            for k, v in optimizer.state[p].items()
        }
        for p in model.parameters()
    ]
    disk_save(state, temporary / f"optimizer-{rank}.pt")
    del state
    disk_save(ema, temporary / f"ema-{rank}.pt")
    disk_save(
        {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(),
        },
        temporary / f"rng-{rank}.pt",
    )


def save_checkpoint(model, optimizer, ema, root, step, cfg, rank):
    temporary = root / "saving"
    temporary.mkdir(exist_ok=True)
    # Serialize rank transfers so CPU copies and filesystem pages do not double-peak.
    for writer in range(dist.get_world_size()):
        if rank == writer:
            save_rank(model, optimizer, ema, temporary, rank)
        dist.barrier()
    if rank == 0:
        (temporary / "metadata.json").write_text(
            json.dumps({"step": step, "config": cfg}, indent=2)
        )
        previous = root / "previous"
        if (root / "latest").exists():
            (root / "latest").rename(previous)
        temporary.rename(root / "latest")
        if previous.exists():
            shutil.rmtree(previous)
        (root / "status.json").write_text(json.dumps({"step": step, "status": "running"}))
    dist.barrier()


def drop_cache(path):
    fd = os.open(path, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)


def load_rank(model, optimizer, root, rank):
    state = torch.load(root / f"model-{rank}.pt", mmap=True, weights_only=False)
    with torch.no_grad():
        for name, value in model.state_dict().items():
            local(value).copy_(state[name])
    del state
    drop_cache(root / f"model-{rank}.pt")
    states = torch.load(root / f"optimizer-{rank}.pt", mmap=True, weights_only=False)
    for p, values in zip(model.parameters(), states, strict=True):
        restored = {}
        for name, value in values.items():
            if name in ("exp_avg", "exp_avg_sq"):
                value = value.to(p.device)
                if isinstance(p, DTensor):
                    value = DTensor.from_local(
                        value, p.device_mesh, p.placements, shape=p.shape, stride=p.stride()
                    )
            elif torch.is_tensor(value):
                # Adam's tiny CPU counters must not retain the checkpoint mmap.
                value = value.clone()
            restored[name] = value
        optimizer.state[p] = restored
    del states, values
    drop_cache(root / f"optimizer-{rank}.pt")
    ema = torch.load(root / f"ema-{rank}.pt", mmap=True, weights_only=False)
    ema = {k: v.clone() for k, v in ema.items()}
    drop_cache(root / f"ema-{rank}.pt")
    rng = torch.load(root / f"rng-{rank}.pt", weights_only=False)
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    torch.cuda.set_rng_state(rng["cuda"])
    return json.loads((root / "metadata.json").read_text())["step"], ema


def load_checkpoint(model, optimizer, root, rank):
    for reader in range(dist.get_world_size()):
        if rank == reader:
            result = load_rank(model, optimizer, root, rank)
        dist.barrier()
    return result


class Indices(Sampler):
    def __init__(self, start, end, rank, world):
        self.start, self.end, self.rank, self.world = start, end, rank, world

    def __iter__(self):
        return iter(range(self.start + self.rank, self.end, self.world))

    def __len__(self):
        return len(range(self.start + self.rank, self.end, self.world))


def move(batch, device):
    return {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()
    }


def clip_grad(model, max_norm):
    gradients = [local(p.grad) for p in model.parameters() if p.grad is not None]
    squared = torch.stack(torch._foreach_norm(gradients)).square().sum()
    dist.all_reduce(squared)
    norm = squared.sqrt()
    torch._foreach_mul_(gradients, (max_norm / (norm + 1e-6)).clamp(max=1))
    return norm


@torch.no_grad()
def validate(model, dataset, rank, world, device, *, sample=False):
    model.eval()
    names = ["Normal", "Low", "ExtremeLow", "Over", "ExtremeOver"]
    totals = torch.zeros((5, 5), device=device)
    # Equal forward counts on all ranks; padded observations have zero metric weight.
    ids = list(range(rank, len(dataset), world))
    count = (len(dataset) + world - 1) // world
    with torch.random.fork_rng(devices=[device.index]):
        for offset in range(count):
            valid = offset < len(ids)
            index = ids[offset] if valid else 0
            batch = move(collate_actions([dataset[index]]), device)
            torch.manual_seed(70000 + index)
            noise = torch.randn_like(batch["actions"])
            tau = model.sample_time(1, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(batch, noise=noise, time=tau)
                prediction = model.sample_sharded(batch, noise) if sample else None
            if valid:
                row = names.index(batch["lighting"][0])
                totals[row, 0] += loss.mean()
                totals[row, 1] += loss[..., :7].mean()
                totals[row, 4] += 1
                if sample:
                    totals[row, 2] += (prediction[..., :7] - batch["actions"][..., :7]).abs().mean()
                    totals[row, 3] += (
                        ((prediction[..., 6] > 0) != (batch["actions"][..., 6] > 0)).float().mean()
                    )
    dist.all_reduce(totals)
    values = {}
    for name, row in zip(names, totals.cpu().tolist(), strict=True):
        values[f"val/{name}/flow32"] = row[0] / row[4]
        values[f"val/{name}/flow7"] = row[1] / row[4]
        if sample:
            values[f"val/{name}/sample_mae7"] = row[2] / row[4]
            values[f"val/{name}/gripper_sign_error"] = row[3] / row[4]
    model.train()
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--initial-eval", action="store_true")
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
    root = args.output
    root.mkdir(exist_ok=True, parents=True)
    model = J5Policy(cfg, initialize=args.resume is None)
    parameter_count = sum(p.numel() for p in model.parameters())
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    pg = model.paligemma_with_expert
    fully_shard(pg.paligemma.language_model.embed_tokens, mp_policy=mp)
    for layer in pg.joint_layers:
        fully_shard(layer, mp_policy=mp)
    # Remaining small parameters, including RAW response, retain FP32 storage/compute policy.
    fully_shard(model)
    register_fsdp_forward_method(model, "sample_sharded")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train["learning_rate"],
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-10,
        foreach=False,
    )
    if args.resume:
        start, ema = load_checkpoint(model, optimizer, args.resume, rank)
    else:
        start = 0
        ema = {k: local(p).detach().cpu().clone() for k, p in model.named_parameters()}
    gc.collect()
    if not args.resume:
        torch.manual_seed(cfg["seed"] + rank)
    sources, _ = load_inventory(cfg["data"]["inventory"])
    sources = [s for s in sources if s.domain == "libero"]
    dataset = RawActionFrames([s for s in sources if s.split == "train"], cfg)
    validation = RawActionFrames(
        [s for s in sources if s.split == "validation"], cfg, validation=True
    )
    loader = DataLoader(
        dataset,
        batch_size=train["micro_batch"],
        sampler=Indices(start * train["batch"], len(dataset), rank, world),
        num_workers=train["workers"],
        prefetch_factor=1,
        multiprocessing_context="spawn",
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_actions,
    )
    iterator = iter(loader)
    tracker = None
    if rank == 0:
        (root / "config.json").write_text(json.dumps(cfg, indent=2))
        (root / "status.json").write_text(json.dumps({"step": start, "status": "running"}))
        (root / "model.json").write_text(json.dumps({"trainable_parameters": parameter_count}))
        if not args.steps:
            import swanlab

            swanlab.login(api_key=Path("/root/.config/owac/swanlab-api-key").read_text().strip())
            tracker = swanlab.init(
                project="OWAC",
                workspace="Jeong",
                experiment_name=cfg["swanlab"]["name"],
                id=cfg["swanlab"]["id"],
                resume="allow",
                config=cfg,
                logdir=str(root / "swanlab"),
                group="J5-action",
                job_type="full-finetune",
            )

    def report(step, metrics):
        if rank == 0:
            record = {"step": step, "metrics": metrics}
            with (root / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            if tracker:
                swanlab.log(metrics, step=step)

    stop = [False]
    signal.signal(signal.SIGUSR1, lambda *_: stop.__setitem__(0, True))
    if args.initial_eval:
        with ema_weights(model, ema):
            report(start, validate(model, validation, rank, world, device))
            selected = [
                next(s for s in validation.sources if s.lighting == light)
                for light in ("Normal", "Low", "ExtremeLow", "Over", "ExtremeOver")
            ]
            probe = RawActionFrames(selected, cfg, validation=True)
            metrics = validate(model, probe, rank, world, device, sample=True)
            report(
                start,
                {
                    key.replace("val/", "probe/initial_sample5/"): value
                    for key, value in metrics.items()
                },
            )
    accumulation = train["batch"] // (world * train["micro_batch"])
    model.train()
    for step in range(start + 1, train["updates"] + 1):
        timer = time.monotonic()
        initial_lr = train["learning_rate"] / (train["warmup"] + 1)
        lr = initial_lr + (train["learning_rate"] - initial_lr) * min(
            (step - 1) / train["warmup"], 1
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        total = torch.zeros(2, device=device)
        for _ in range(accumulation):
            batch = move(next(iterator), device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses = model(batch)
                loss = losses.mean()
            (loss / accumulation).backward()
            total += torch.stack((loss.detach(), losses[..., :7].mean().detach())) / accumulation
        grad = clip_grad(model, 1.0)
        optimizer.step()
        with torch.no_grad():
            for name, p in model.named_parameters():
                ema[name].lerp_(local(p).detach().cpu(), 1 - train["ema_decay"])
        dist.all_reduce(total)
        elapsed = time.monotonic() - timer
        if step == start + 1 or step % train["log_every"] == 0:
            mem = int(Path("/sys/fs/cgroup/memory.current").read_text()) / 2**30
            memory_stat = dict(
                line.split() for line in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines()
            )
            events = dict(
                line.split()
                for line in Path("/sys/fs/cgroup/memory.events").read_text().splitlines()
            )
            report(
                step,
                {
                    "train/flow32": total[0].item() / world,
                    "train/flow7": total[1].item() / world,
                    "train/grad_norm": grad.item(),
                    "train/lr": lr,
                    "train/seconds_per_update": elapsed,
                    "train/images_per_second": train["batch"] / elapsed,
                    "host/memory_gib": mem,
                    "host/anonymous_gib": int(memory_stat["anon"]) / 2**30,
                    "host/file_cache_gib": int(memory_stat["file"]) / 2**30,
                    "host/shared_gib": int(memory_stat["shmem"]) / 2**30,
                    "host/oom_kills": int(events["oom_kill"]),
                    "gpu/peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                },
            )
        stopping = torch.tensor(int(stop[0]), device=device)
        dist.all_reduce(stopping, op=dist.ReduceOp.MAX)
        stop[0] = bool(stopping.item())
        final = step == train["updates"] or (args.steps is not None and step - start >= args.steps)
        if step == start + 1 or step % train["save_every"] == 0 or final or stop[0]:
            save_checkpoint(model, optimizer, ema, root, step, cfg, rank)
        if not args.steps and (step % train["eval_every"] == 0 or final):
            with ema_weights(model, ema):
                report(
                    step,
                    validate(
                        model,
                        validation,
                        rank,
                        world,
                        device,
                        sample=step % train["sample_every"] == 0 or final,
                    ),
                )
        if final or stop[0]:
            break
    if rank == 0:
        (root / "status.json").write_text(
            json.dumps({"step": step, "status": "paused" if stop[0] else "completed"})
        )
        if tracker:
            swanlab.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
