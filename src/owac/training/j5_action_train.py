"""Full pi05 fine-tuning: official flow objective/settings, streamed RAW, FSDP2."""

import argparse
import gc
import json
import math
import os
import random
import shutil
import signal
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard, register_fsdp_forward_method
from torch.distributed.tensor import DTensor
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler

from owac.raw.all_bench_distillation import load_inventory
from owac.training import j5_monitoring
from owac.training.j5_action_data import RawActionFrames, collate_actions
from owac.training.j5_action_metrics import action_metrics, action_sums
from owac.training.j5_pi05 import J5Policy


def local(value):
    return value.to_local() if isinstance(value, DTensor) else value


@contextmanager
def ema_weights(model, ema, *, sharded=True):
    parameters = {name: p for name, p in model.named_parameters() if name in ema}
    online = {
        name: local(p).detach().cpu() if sharded else p.detach().clone()
        for name, p in parameters.items()
    }
    with torch.no_grad():
        for name, p in parameters.items():
            local(p).copy_(ema[name])
    try:
        yield
    finally:
        # Root weights stay gathered after no-grad forwards (no backward hook).
        if sharded:
            model.reshard()
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in online:
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
    action_validation = dataset.cfg.get("validation", {}).get("action_metrics", False)
    totals = torch.zeros((5, 13), device=device, dtype=torch.float64)
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
                if sample and not hasattr(model, "reshard"):
                    from owac.training.j5_pi05 import observation

                    prediction = model.sample_actions(
                        device, observation(batch), noise=noise, num_steps=10
                    )
                else:
                    prediction = model.sample_sharded(batch, noise) if sample else None
            if valid:
                row = names.index(batch["lighting"][0])
                totals[row, 0] += loss.mean()
                totals[row, 1] += loss[..., :7].mean()
                totals[row, 4] += 1
                if sample:
                    if action_validation:
                        totals[row, 5:] += action_sums(
                            prediction,
                            batch["actions"],
                            batch["action_valid"],
                            dataset.stats["actions"],
                            dataset.cfg["closed_loop"]["execute_horizon"],
                        )
                    totals[row, 2] += (prediction[..., :7] - batch["actions"][..., :7]).abs().mean()
                    totals[row, 3] += (
                        ((prediction[..., 6] > 0) != (batch["actions"][..., 6] > 0)).float().mean()
                    )
    dist.all_reduce(totals)
    if action_validation:
        aggregate = totals.sum(0).cpu().tolist()
        values = {"objective/validation_flow32": aggregate[0] / aggregate[4]}
        if sample:
            values.update(action_metrics(aggregate[5:]))
            values["validation/anchors"] = aggregate[4]
            if (totals[:, 4] > 0).sum().item() > 1:
                for name, row in zip(names, totals.cpu().tolist(), strict=True):
                    if row[4]:
                        values[f"validation/{name}_action_mse"] = row[5] / (7 * row[11])
        model.train()
        return values
    values = {}
    for name, row in zip(names, totals.cpu().tolist(), strict=True):
        if row[4] == 0:
            continue
        values[f"val/{name}/flow32"] = row[0] / row[4]
        values[f"val/{name}/flow7"] = row[1] / row[4]
        if sample:
            values[f"val/{name}/sample_mae7"] = row[2] / row[4]
            values[f"val/{name}/gripper_sign_error"] = row[3] / row[4]
    if sample:
        values["val/sample_mae7"] = (totals[:, 2].sum() / totals[:, 4].sum()).item()
    model.train()
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--initial-eval", action="store_true")
    parser.add_argument("--until-step", type=int)
    parser.add_argument("--reset-patience", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    train = cfg["training"]
    lora = cfg.get("adaptation", {}).get("kind") == "lora"
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
    if lora:
        from owac.training import j5_lora

        # Sequential CPU construction bounds temporary base-checkpoint memory.
        for reader in range(world):
            if reader == rank:
                model = j5_lora.build_policy(cfg).to(device)
            dist.barrier()
    else:
        model = J5Policy(cfg, initialize=args.resume is None)
        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        pg = model.paligemma_with_expert
        fully_shard(pg.paligemma.language_model.embed_tokens, mp_policy=mp)
        for layer in pg.joint_layers:
            fully_shard(layer, mp_policy=mp)
        fully_shard(model)
        register_fsdp_forward_method(model, "sample_sharded")
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    parameters = [p for p in model.parameters() if p.requires_grad]
    parameter_groups = parameters
    if "group_learning_rates" in train:
        parameter_groups = j5_lora.parameter_groups(model, train)
    elif "expert_learning_rate" in train:
        parameter_groups = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if p.requires_grad and n.startswith("raw_projector.")
                ],
                "lr_scale": 1.0,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if p.requires_grad and not n.startswith("raw_projector.")
                ],
                "lr_scale": train["expert_learning_rate"] / train["learning_rate"],
            },
        ]
    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=train["learning_rate"],
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-10,
        foreach=lora,
    )
    if args.resume:
        restore = j5_lora.load if lora else load_checkpoint
        start, ema = restore(model, optimizer, args.resume, rank)
    else:
        start = cfg["adaptation"]["initial_step"] if lora else 0
        ema = {
            k: p.detach().clone() if lora else local(p).detach().cpu().clone()
            for k, p in model.named_parameters()
            if p.requires_grad
        }
    selection = {"best": None, "bad_evals": 0, "lr_factor": 1.0}
    if args.resume and lora:
        selection.update(
            torch.load(args.resume, map_location="cpu", weights_only=False)
            .get("config", {})
            .get("selection", {})
        )
    if args.reset_patience:
        selection["bad_evals"] = 0
    selection_metric = train.get("selection_metric", "val/sample_mae7")
    if selection.get("metric", "val/sample_mae7") != selection_metric:
        selection = {"best": None, "bad_evals": 0, "lr_factor": selection["lr_factor"]}
    selection["metric"] = selection_metric
    cfg["selection"] = selection
    core = model
    if lora:
        # The last language output is unused by the action loss, including its LoRA.
        model = DistributedDataParallel(
            core,
            device_ids=[device.index],
            broadcast_buffers=False,
            find_unused_parameters=cfg["adaptation"]["ranks"][0] > 0,
            gradient_as_bucket_view=True,
        )
    gc.collect()
    if not args.resume:
        torch.manual_seed(cfg["seed"] + rank)
    sources, _ = load_inventory(cfg["data"]["inventory"])
    sources = [s for s in sources if s.domain == "libero"]
    if cfg["data"].get("lighting"):
        sources = [s for s in sources if s.lighting in cfg["data"]["lighting"]]
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
        generator=torch.Generator().manual_seed(cfg["seed"] + rank),
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
            if cfg.get("monitoring", {}).get("version") == 2:
                swanlab.merge_settings(
                    swanlab.Settings(probe=swanlab.Settings.Probe(monitor=False))
                )
            tracker = swanlab.init(
                project=cfg["swanlab"].get("project", "OWAC"),
                workspace="Jeong",
                experiment_name=cfg["swanlab"]["name"],
                id=cfg["swanlab"]["id"],
                resume="allow",
                config=cfg,
                logdir=str(root / "swanlab"),
                group="J5-action",
                job_type="lora-finetune" if lora else "full-finetune",
            )

            if cfg.get("monitoring", {}).get("version") == 2:
                j5_monitoring.setup(swanlab, cfg, root)

    def report(step, metrics):
        if cfg.get("monitoring", {}).get("version") == 2:
            metrics = j5_monitoring.canonical(metrics)
        if rank == 0:
            record = {"step": step, "metrics": metrics}
            with (root / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            if tracker:
                swanlab.log(metrics, step=step)

    stop = [False]
    signal.signal(signal.SIGUSR1, lambda *_: stop.__setitem__(0, True))
    if args.initial_eval and cfg.get("validation", {}).get("action_metrics"):
        if selection["best"] is None:
            with ema_weights(core, ema, sharded=not lora):
                metrics = validate(core, validation, rank, world, device, sample=True)
            report(start, metrics)
            if train.get("retain_best"):
                selection.update(best=metrics[selection_metric], best_step=start, bad_evals=0)
                j5_lora.save(core, optimizer, ema, root, start, cfg, rank)
                if rank == 0:
                    (root / "best.pt").unlink(missing_ok=True)
                    os.link(root / "latest.pt", root / "best.pt")
                    (root / "selection.json").write_text(json.dumps(selection, indent=2))
    elif args.initial_eval:
        with ema_weights(core, ema, sharded=not lora):
            report(start, validate(core, validation, rank, world, device))
            selected = [
                next(s for s in validation.sources if s.lighting == light)
                for light in dict.fromkeys(s.lighting for s in validation.sources)
            ]
            probe = RawActionFrames(selected, cfg, validation=True)
            metrics = validate(core, probe, rank, world, device, sample=True)
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
        if train.get("schedule") == "cosine":
            progress = max(0, step - train["warmup"]) / (train["updates"] - train["warmup"])
            lr *= 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2
            lr *= selection["lr_factor"]
        for group in optimizer.param_groups:
            group["lr"] = lr * group.get("lr_scale", 1.0)
        optimizer.zero_grad(set_to_none=True)
        total = torch.zeros(2, device=device)
        for micro in range(accumulation):
            batch = move(next(iterator), device)
            sync = model.no_sync() if lora and micro < accumulation - 1 else nullcontext()
            with sync:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    losses = model(batch)
                    loss = losses.mean()
                (loss / accumulation).backward()
            total += torch.stack((loss.detach(), losses[..., :7].mean().detach())) / accumulation
        grad = torch.nn.utils.clip_grad_norm_(parameters, 1.0) if lora else clip_grad(model, 1.0)
        optimizer.step()
        with torch.no_grad():
            if lora:
                torch._foreach_lerp_(list(ema.values()), parameters, 1 - train["ema_decay"])
            else:
                for name, p in model.named_parameters():
                    ema[name].lerp_(local(p).detach().cpu(), 1 - train["ema_decay"])
        dist.all_reduce(total)
        # GPU-resident LoRA EMA is asynchronous; include it in wall-clock timing.
        torch.cuda.synchronize()
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
                    "train/validation_lr_factor": selection["lr_factor"],
                    **{
                        f"train/lr_{g['name']}": g["lr"]
                        for g in optimizer.param_groups
                        if "name" in g
                    },
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
        final = (
            step == train["updates"]
            or step == args.until_step
            or (args.steps is not None and step - start >= args.steps)
        )
        if step == start + 1 or step % train["save_every"] == 0 or final or stop[0]:
            save = j5_lora.save if lora else save_checkpoint
            save(core, optimizer, ema, root, step, cfg, rank)
            if rank == 0:
                (root / "status.json").write_text(json.dumps({"step": step, "status": "running"}))
        if (not args.steps or train.get("retain_best")) and (
            step % train["eval_every"] == 0 or final
        ):
            with ema_weights(core, ema, sharded=not lora):
                metrics = validate(
                    core,
                    validation,
                    rank,
                    world,
                    device,
                    sample=step % train["sample_every"] == 0 or final,
                )
                report(step, metrics)
            if train.get("retain_best") and selection_metric in metrics:
                improved = (
                    selection["best"] is None or metrics[selection_metric] < selection["best"]
                )
                if improved:
                    selection["best"] = metrics[selection_metric]
                    selection["best_step"] = step
                    selection["bad_evals"] = 0
                else:
                    selection["bad_evals"] += 1
                    if selection["bad_evals"] >= train["plateau_patience"]:
                        selection["lr_factor"] *= 0.5
                        selection["bad_evals"] = 0
                        # Restore a coherent model/Adam pair; keep the current data cursor.
                        best = torch.load(root / "best.pt", map_location="cpu", weights_only=False)
                        j5_lora.copy_trainable(core, best["model"])
                        optimizer.load_state_dict(best["optimizer"])
                        ema = {name: value.to(device) for name, value in best["ema"].items()}
                        del best
                        report(
                            step,
                            {
                                "train/rollback_to_step": selection["best_step"],
                                "train/validation_lr_factor": selection["lr_factor"],
                            },
                        )
                j5_lora.save(core, optimizer, ema, root, step, cfg, rank)
                if rank == 0:
                    if improved:
                        (root / "best.pt").unlink(missing_ok=True)
                        os.link(root / "latest.pt", root / "best.pt")
                    if step % train["retain_every"] == 0 or final:
                        target = root / f"step-{step:06d}.pt"
                        target.unlink(missing_ok=True)
                        os.link(root / "latest.pt", target)
                    (root / "selection.json").write_text(json.dumps(selection, indent=2))
        if final or stop[0]:
            break
    if rank == 0:
        (root / "status.json").write_text(
            json.dumps(
                {
                    "step": step,
                    "status": (
                        "paused"
                        if stop[0]
                        else "segment_completed"
                        if args.until_step and step < train["updates"]
                        else "completed"
                    ),
                }
            )
        )
        if tracker:
            swanlab.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
