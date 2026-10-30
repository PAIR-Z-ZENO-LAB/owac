"""Distributed RAW-only DINO/iBOT training with complete resumable checkpoints."""

import argparse
import copy
import json
import math
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from owac.raw.all_bench_distillation import load_inventory
from owac.training.raw_exposure import dense_consistency
from owac.training.raw_relations import (
    diversity_loss,
    relation_loss,
    temporal_loss,
    temporal_targets,
)
from owac.training.raw_ssl import RawEncoder, RawSSL, objective, sinkhorn
from owac.training.raw_ssl_data import RawViews, ResumeSampler, block_masks
from owac.training.raw_ssl_eval import evaluate


def atomic_save(value, path):
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    temp.replace(path)


def rng_state():
    return dict(
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state(),
        python=random.getstate(),
        numpy=np.random.get_state(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--steps", type=int, help="Explicit engineering run length; schedule unchanged"
    )
    parser.add_argument("--no-tracking", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", device_id=device)
    train = cfg["training"]
    torch.set_num_threads(train["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"] + rank)
    np.random.seed(cfg["seed"] + rank)
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (root / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
    student = RawSSL(cfg).to(device)
    teacher = copy.deepcopy(student).eval().requires_grad_(False)
    groups = []
    for name, parameter in student.named_parameters():
        decay = parameter.ndim > 1 and "log_pivot" not in name
        groups.append(
            dict(
                params=[parameter],
                weight_decay=0.04 if decay else 0.0,
                decay=decay,
                lr_multiplier=0.2 if "patch_embeddings" in name else 1.0,
            )
        )
    optimizer = torch.optim.AdamW(groups, lr=train["lr"], betas=(0.9, 0.999), fused=True)
    start = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False, mmap=True)
        student.load_state_dict(checkpoint["student"])
        teacher.load_state_dict(checkpoint["teacher"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start = checkpoint["step"]
        state = checkpoint["rng"][rank]
        torch.set_rng_state(state["torch"])
        torch.cuda.set_rng_state(state["cuda"])
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and value.device.type == "cpu":
                    state[key] = value.clone()
        del checkpoint
    elif train.get("initialize_from"):
        checkpoint = torch.load(
            train["initialize_from"], map_location="cpu", weights_only=False, mmap=True
        )
        # New optimization schedule, initialized only from our own RAW-trained EMA.
        student.load_state_dict(checkpoint["teacher"])
        teacher.load_state_dict(checkpoint["teacher"])
        del checkpoint
    anchor = None
    if "relations" in cfg:
        anchor = RawEncoder(cfg["model"]).to(device).eval().requires_grad_(False)
        reference = torch.load(
            cfg["relations"]["anchor"], map_location="cpu", weights_only=False, mmap=True
        )
        anchor.load_state_dict(reference["encoder"])
        del reference
    ddp = DDP(student, device_ids=[rank])
    sources, _ = load_inventory(cfg["data"]["inventory"])
    sources = [s for s in sources if s.domain in cfg["data"]["domains"]]
    training = [s for s in sources if s.split == "train"]
    validation = [s for s in sources if s.split == "validation"]
    dataset = RawViews(training, cfg)
    sampler = ResumeSampler(start, train["updates"], train["batch"], rank, world)
    loader = DataLoader(
        dataset,
        batch_size=train["batch"] // world,
        sampler=sampler,
        num_workers=train["workers"],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=train["prefetch_factor"],
        generator=torch.Generator().manual_seed(cfg["seed"] + rank),
    )
    tracker = None
    if rank == 0 and not args.no_tracking:
        import swanlab

        swanlab.login(
            api_key=Path("/root/.config/owac/swanlab-api-key").read_text().strip(), save=False
        )
        tracking = cfg["tracking"]
        tracker = swanlab.init(
            project=tracking["project"],
            name=tracking["name"],
            id=tracking["run_id"],
            group=tracking["group"],
            job_type=tracking["job_type"],
            resume="allow",
            public=False,
            config=cfg,
            logdir=str(root / "swanlab"),
        )
        (root / "swanlab.json").write_text(json.dumps({"url": tracker.url}))
    stop = [False]
    signal.signal(signal.SIGUSR1, lambda *_: stop.__setitem__(0, True))

    def save(step):
        states = [None] * world
        dist.all_gather_object(states, rng_state())
        if rank == 0:
            atomic_save(
                dict(
                    student=student.state_dict(),
                    teacher=teacher.state_dict(),
                    optimizer=optimizer.state_dict(),
                    step=step,
                    rng=states,
                    config=cfg,
                ),
                root / "latest.pt",
            )
            atomic_save(
                dict(encoder=teacher.encoder.state_dict(), config=cfg, step=step),
                root / "encoder.pt",
            )
        dist.barrier()

    def report(step, metrics, paths=()):
        if rank != 0:
            return
        with (root / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, "metrics": metrics}) + "\n")
        print(json.dumps({"step": step, **metrics}), flush=True)
        if tracker:
            import swanlab

            values = dict(metrics)
            if paths:
                values["eval/token-and-query"] = [swanlab.Image(p) for p in paths]
            swanlab.log(values, step=step)

    if start == 0 and not args.steps:
        if rank == 0:
            values, paths = evaluate(teacher.encoder, validation, cfg, root, 0, device)
            report(0, values, paths)
        dist.barrier()
    timer = time.monotonic()
    last_log_step = start
    for step, views in enumerate(loader, start + 1):
        global_views, local_views = views[:2]
        batch = global_views.shape[0]
        global_views = global_views.transpose(0, 1).flatten(0, 1).to(device, non_blocking=True)
        local_views = local_views.transpose(0, 1).flatten(0, 1).to(device, non_blocking=True)
        student_views = (
            views[2].transpose(0, 1).flatten(0, 1).to(device, non_blocking=True)
            if "exposure" in cfg
            else global_views
        )
        masks = block_masks(
            2 * batch, cfg["crops"]["global_size"] // cfg["model"]["patch_size"], device
        )
        progress = step / train["updates"]
        cosine = (1 + math.cos(math.pi * progress)) / 2
        lr = (
            train["lr"] * min(step / train["warmup"], 1.0)
            if step < train["warmup"]
            else (
                train["min_lr"]
                + (train["lr"] - train["min_lr"])
                * (
                    1
                    + math.cos(
                        math.pi * (step - train["warmup"]) / (train["updates"] - train["warmup"])
                    )
                )
                / 2
            )
        )
        for group in optimizer.param_groups:
            group["lr"] = lr * group["lr_multiplier"]
            if group["decay"]:
                group["weight_decay"] = (
                    train["weight_decay_end"]
                    + (train["weight_decay_start"] - train["weight_decay_end"]) * cosine
                )
        temperature = cfg["loss"]["teacher_temp_start"] + (
            cfg["loss"]["teacher_temp_end"] - cfg["loss"]["teacher_temp_start"]
        ) * min(step / train["warmup"], 1)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            target = teacher(global_views)
            target_patch = teacher.patch_head(target["patches"][masks])
        if anchor is not None:
            normal = views[4].to(device).repeat(2)
            past = views[3].to(device, non_blocking=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                _, previous = teacher.encoder(past)
                # Only well-exposed RAW supplies the frozen spatial relation reference.
                anchor_features = anchor(global_views[normal])[1] if views[4].any() else None
            temporal_target, confidence, displacement = temporal_targets(
                target["patches"][:batch],
                previous,
                cfg["relations"]["match_radius"],
                cfg["relations"]["match_temperature"],
            )
        tg, tp = sinkhorn(target["global"], temperature), sinkhorn(target_patch, temperature)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = ddp(student_views, local_views, masks)
            loss, metrics = objective(
                result, tg, tp, batch, cfg["crops"]["local_count"], cfg["loss"]
            )
            if "exposure" in cfg:
                paired = dense_consistency(result["patches"], target["patches"], masks)
                loss = loss + cfg["exposure"]["dense_weight"] * paired
                metrics["exposure_dense"] = paired.detach()
            if anchor is not None:
                settings = cfg["relations"]
                spatial = (
                    relation_loss(result["patches"][normal], anchor_features, masks[normal])
                    if anchor_features is not None
                    else result["patches"].sum() * 0
                )
                variance, covariance = diversity_loss(result["patches"], masks)
                temporal = temporal_loss(
                    result["patches"][:batch], temporal_target, confidence, masks[:batch]
                )
                temporal_weight = settings["temporal_weight"] * min(
                    step / settings["temporal_warmup"], 1
                )
                loss = (
                    loss
                    + settings["spatial_weight"] * spatial
                    + settings["variance_weight"] * variance
                    + settings["covariance_weight"] * covariance
                    + temporal_weight * temporal
                )
                metrics.update(
                    spatial_relation=spatial.detach(),
                    token_variance=variance.detach(),
                    token_covariance=covariance.detach(),
                    temporal=temporal.detach(),
                    temporal_confidence=confidence.mean(),
                    temporal_displacement=displacement,
                )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 3.0, error_if_nonfinite=True)
        if step <= train["freeze_prototypes"]:
            student.global_head.weight.grad = None
            student.patch_head.weight.grad = None
        optimizer.step()
        momentum = 1 - (1 - train["ema_start"]) * cosine
        with torch.no_grad():
            torch._foreach_mul_(list(teacher.parameters()), momentum)
            torch._foreach_add_(
                list(teacher.parameters()), list(student.parameters()), alpha=1 - momentum
            )
        if step % train["log_every"] == 0 or step == start + 1:
            elapsed = time.monotonic() - timer
            count = step - last_log_step
            last_log_step = step
            values = {f"train/{k}": float(v) for k, v in metrics.items()}
            values.update(
                {
                    "train/loss": float(loss.detach()),
                    "train/lr": lr,
                    "train/images_per_second": count * train["batch"] / elapsed,
                    "train/step_seconds": elapsed / count,
                    "gpu/allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                }
            )
            if rank == 0:
                stat = dict(
                    line.split()
                    for line in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines()
                )
                events = dict(
                    line.split()
                    for line in Path("/sys/fs/cgroup/memory.events").read_text().splitlines()
                )
                values.update(
                    {
                        "host/anonymous_gib": int(stat["anon"]) / 2**30,
                        "host/shared_memory_gib": int(stat["shmem"]) / 2**30,
                        "host/file_cache_gib": (int(stat["file"]) - int(stat["shmem"])) / 2**30,
                        "host/memory_current_gib": int(
                            Path("/sys/fs/cgroup/memory.current").read_text()
                        )
                        / 2**30,
                        "host/oom_kills": int(events["oom_kill"]),
                    }
                )
            report(step, values)
            timer = time.monotonic()
        if step % train["log_every"] == 0:
            should_stop = torch.tensor(int(stop[0]), device=device)
            dist.all_reduce(should_stop, op=dist.ReduceOp.MAX)
            stop[0] = bool(should_stop.item())
        final = step == train["updates"] or (args.steps is not None and step - start >= args.steps)
        if step % train["save_every"] == 0 or final or stop[0]:
            save(step)
        if (step % train["eval_every"] == 0 or final) and not args.steps:
            if rank == 0:
                values, paths = evaluate(teacher.encoder, validation, cfg, root, step, device)
                report(step, values, paths)
            dist.barrier()
            timer = time.monotonic()
        if final or stop[0]:
            break
    if rank == 0:
        (root / "status.json").write_text(
            json.dumps({"step": step, "status": "paused" if stop[0] else "completed"})
        )
        if tracker:
            import swanlab

            swanlab.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
