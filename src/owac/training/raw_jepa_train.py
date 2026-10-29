"""Distributed RAW JEPA training and fixed held-source validation."""

from __future__ import annotations

import json
import math
import os
import random
import signal
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from owac.raw.all_bench_distillation import EpisodeCache, _restore_rng, _rng_state, load_inventory
from owac.training.raw_dense_refinement import (
    dense_objective,
    fixed_spatial_anchor,
    overlapping_crops,
)
from owac.training.raw_jepa import (
    block_mask,
    build_jepa,
    prediction_loss,
    representation_metrics,
    update_target,
)


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def emit(root, step, metrics):
    with (root / "metrics.jsonl").open("a") as stream:
        stream.write(json.dumps({"step": step, "metrics": metrics}) + "\n")


class RawWindows:
    def __init__(self, sources, cfg):
        self.sources = {s.group_id: s for s in sources if s.domain in cfg["data"]["domains"]}
        self.cache = EpisodeCache(self.sources, cfg["training"]["working_cache_sources"])
        self.cfg = cfg["data"]

    def keys(self, split):
        return [
            (s.group_id, e)
            for s in self.sources.values()
            if s.split == split
            for e in range(self.cfg["minimum_endpoint"], s.num_frames, self.cfg["endpoint_stride"])
        ]

    def validation_keys(self):
        result = []
        for source in self.sources.values():
            if source.split != "validation":
                continue
            start, end = self.cfg["minimum_endpoint"], source.num_frames - 1
            result.extend((source.group_id, e) for e in sorted({start, (start + end) // 2, end}))
        return result

    def native_batch(self, keys, device):
        # Only raw numeric members are read; no actions, prompts, target caches or RGB.
        raw = torch.stack([self.cache.get(s).burst(e) for s, e in keys]).to(device)
        # Preserve camera ordering, frame ordering and full field of view.
        return raw.flatten(0, 1)

    def batch(self, keys, device):
        raw = self.native_batch(keys, device)
        return F.interpolate(
            raw.flatten(0, 1), size=(224, 224), mode="bilinear", align_corners=False
        ).reshape(len(keys) * 2, 6, 3, 224, 224)


def epoch_batches(keys, batch_size, seed, epoch):
    rng = random.Random(seed + epoch)
    sources = defaultdict(list)
    for key in keys:
        sources[key[0]].append(key)
    chunks = []
    for values in sources.values():
        rng.shuffle(values)
        chunks.extend(values[i : i + 4] for i in range(0, len(values), 4))
    rng.shuffle(chunks)
    flat = [key for chunk in chunks for key in chunk]
    # Explicit epoch-tail wrapping keeps the DDP batch equal across ranks.
    padding = (-len(flat)) % batch_size
    flat.extend(flat[:padding])
    return [flat[i : i + batch_size] for i in range(0, len(flat), batch_size)]


def lr_factor(step, settings):
    warmup = settings["warmup_updates"]
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / (settings["updates"] - warmup)
    floor = settings["final_lr_ratio"]
    return floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2


@torch.no_grad()
def validate(online, target, dataset, config, device, rank, world, step, anchor=None):
    online.eval()
    rows = []
    keys = dataset.validation_keys()
    for index in range(rank, len(keys), world):
        key = keys[index]
        raw = dataset.batch([key], device)
        mask = block_mask(
            16,
            random.Random(1000000 + index),
            **{k: config["mask"][k] for k in ("blocks", "scale")},
        ).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            full = target(raw)
            context = online.encoder(raw, mask)
            prediction = online.predictor(context, mask)
            # Camera-swapped context: same positions/predictor, wrong image evidence.
            swapped = online.predictor(context.flip(0), mask)
        loss, targets = prediction_loss(prediction, full, mask)
        values = {k: float(v) for k, v in representation_metrics(prediction, targets, full).items()}
        values["prediction_loss"] = float(loss)
        values["swapped_context_loss"] = float(F.smooth_l1_loss(swapped.float(), targets))
        # Same targets for both baselines, making loss curves directly comparable.
        normalized = F.layer_norm(full.float(), (full.shape[-1],))
        mean = normalized[:, ~mask].mean(1, keepdim=True).expand_as(targets)
        values["visible_mean_loss"] = float(F.smooth_l1_loss(mean, targets))
        values["masked_fraction"] = float(mask.float().mean())
        if anchor is not None:
            views, ia, ib = overlapping_crops(
                dataset.native_batch([key], device), random.Random(2000000 + index)
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                a, b = (online.encoder(view) for view in views)
                aa, ab = (anchor(view) for view in views)
            ce, gram, matches = dense_objective(
                a, b, aa, ab, ia, ib, temperature=config["dense"]["temperature"]
            )
            values.update(
                dense_correspondence_loss=float(ce),
                gram_anchor_loss=float(gram),
                dense_exact_fraction=float(matches),
            )
        rows.append((dataset.sources[key[0]].lighting, values, normalized.mean(1).cpu()))
    gathered = [None] * world
    if world > 1:
        dist.all_gather_object(gathered, rows)
        rows = [row for rank_rows in gathered for row in rank_rows]
    groups = defaultdict(list)
    for lighting, values, _pooled in rows:
        groups["all"].append(values)
        groups[lighting].append(values)
    metrics = {}
    for group, values in groups.items():
        for name in values[0]:
            metrics[f"validation/{group}/{name}"] = sum(v[name] for v in values) / len(values)
        metrics[f"validation/{group}/windows"] = len(values)
    pooled = torch.cat([row[2] for row in rows])
    metrics["representation/validation_between_image_std"] = float(
        pooled.std(0, correction=0).mean()
    )
    metrics["representation/validation_between_image_cosine"] = float(
        F.cosine_similarity(pooled, pooled.roll(2, 0), dim=-1).mean()
    )
    metrics["validation/all/context_advantage"] = (
        metrics["validation/all/swapped_context_loss"] - metrics["validation/all/prediction_loss"]
    )
    online.train()
    return metrics


class BudgetStream:
    def __init__(self, stream, allowance):
        self.stream, self.allowance = stream, allowance

    def write(self, data):
        if self.stream.tell() + len(data) > self.allowance:
            raise OSError("JEPA checkpoint exceeds the configured output budget")
        return self.stream.write(data)

    def flush(self):
        self.stream.flush()


def save_file(path, payload, root, storage):
    # Only count this run, using the one-time baseline for pre-existing assets.
    atomic = storage.get("atomic_checkpoints", False)
    destination = path.with_suffix(path.suffix + ".tmp") if atomic else path
    others = sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and p != destination)
    allowance = storage["cap_bytes"] - storage["existing_owned_bytes"] - others
    with destination.open("wb") as stream:
        torch.save(payload, BudgetStream(stream, allowance))
        stream.flush()
        os.fsync(stream.fileno())
    if atomic:
        destination.replace(path)


def train(config, output: Path, resume: Path | None = None):
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    torch.set_num_threads(4)
    if world > 1:
        dist.init_process_group("nccl")
    settings = config["training"]
    torch.manual_seed(settings["seed"])
    random.seed(settings["seed"])
    sources, _ = load_inventory(config["data"]["inventory"])
    dataset = RawWindows(sources, config)
    keys = dataset.keys("train")
    online, target, projector = build_jepa(config, device)
    anchor = (
        fixed_spatial_anchor(online.encoder, config["student"]["refinement_layer"])
        if config.get("dense")
        else None
    )
    groups = [
        {
            "params": [
                p
                for n, p in online.encoder.named_parameters()
                if "patch_embedding" in n or "gate_logit" in n
            ],
            "lr": settings["stem_lr"],
            "name": "stem",
        },
        {
            "params": [
                p
                for n, p in online.encoder.named_parameters()
                if "patch_embedding" not in n and "gate_logit" not in n
            ],
            "lr": settings["backbone_lr"],
            "name": "backbone",
        },
        {
            "params": list(online.predictor.parameters()),
            "lr": settings["predictor_lr"],
            "name": "predictor",
        },
    ]
    optimizer = torch.optim.AdamW(
        groups, betas=tuple(settings["betas"]), weight_decay=settings["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_factor(s, settings))
    model = DDP(online, device_ids=[local]) if world > 1 else online
    output.mkdir(parents=True, exist_ok=True)
    step, epoch, cursor = 0, 0, 0
    if resume:
        payload = torch.load(resume, map_location="cpu", weights_only=True)
        online.load_state_dict(payload["online"])
        target.load_state_dict(payload["target"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        step, epoch, cursor = payload["step"], payload["epoch"], payload["cursor"]
        _restore_rng(payload["rng_by_rank"][rank], local)
        del payload
    else:
        torch.manual_seed(settings["seed"] + rank)
        random.seed(settings["seed"] + rank)
    paused = False

    def pause(signum, frame):
        nonlocal paused
        paused = True

    signal.signal(signal.SIGUSR1, pause)
    params = list(online.parameters())

    def save():
        states = [None] * world
        if world > 1:
            dist.all_gather_object(states, _rng_state(local))
        else:
            states[0] = _rng_state(local)
        if rank == 0:
            # Budget fits one full state plus one inference encoder, not two full states.
            # Preserve an encoder recovery file before replacing the full state in place.
            # A false completion receipt must never be used for exact-state resume.
            export = output / "encoder.pt"
            if not config["storage"].get("atomic_checkpoints", False):
                export.unlink(missing_ok=True)
            save_file(
                export,
                {
                    "encoder": online.encoder.state_dict(),
                    "projector": projector.state_dict(),
                    "step": step,
                    "config": config,
                },
                output,
                config["storage"],
            )
            write_json(
                output / "checkpoint.json",
                {"complete": False, "step": step, "encoder_complete": True},
            )
            save_file(
                output / "latest.pt",
                {
                    "online": online.state_dict(),
                    "target": target.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step,
                    "epoch": epoch,
                    "cursor": cursor,
                    "rng_by_rank": states,
                    "config": config,
                },
                output,
                config["storage"],
            )
            write_json(
                output / "checkpoint.json",
                {"complete": True, "step": step, "encoder_complete": True},
            )
        if world > 1:
            dist.barrier()

    if rank == 0:
        write_json(output / "config.json", config)
        write_json(
            output / "data-summary.json",
            {
                "source_split": dict(Counter(s.split for s in dataset.sources.values())),
                "train_windows": len(keys),
                "validation_windows": len(dataset.validation_keys()),
                "train_lighting": dict(
                    Counter(s.lighting for s in dataset.sources.values() if s.split == "train")
                ),
                "encoder_parameters": sum(p.numel() for p in online.encoder.parameters()),
                "predictor_parameters": sum(p.numel() for p in online.predictor.parameters()),
                "retained_projector_parameters": sum(p.numel() for p in projector.parameters()),
                "external_raw_teacher": False,
                "action_backend": False,
            },
        )
        write_json(
            output / "status.json",
            {"status": "validating", "step": step, "target": settings["updates"]},
        )
    if resume is None:
        metrics = validate(online, target, dataset, config, device, rank, world, step, anchor)
        if rank == 0:
            emit(output, step, metrics)
    while step < settings["updates"]:
        batches = epoch_batches(keys, settings["effective_batch"], settings["seed"], epoch)
        for index in range(cursor, len(batches)):
            started = time.monotonic()
            mine = batches[index][rank::world]
            micro = settings["micro_batch_per_rank"]
            optimizer.zero_grad(set_to_none=True)
            metrics_sum = defaultdict(float)
            for offset in range(0, len(mine), micro):
                batch_keys = mine[offset : offset + micro]
                raw = dataset.batch(batch_keys, device)
                mask = block_mask(
                    16, random, **{k: config["mask"][k] for k in ("blocks", "scale")}
                ).to(device)
                sync = (
                    nullcontext() if world == 1 or offset + micro >= len(mine) else model.no_sync()
                )
                with sync, torch.autocast("cuda", dtype=torch.bfloat16):
                    with torch.no_grad():
                        full = target(raw)
                    if anchor is not None:
                        views, ia, ib = overlapping_crops(
                            dataset.native_batch(batch_keys, device), random
                        )
                        with torch.no_grad():
                            aa, ab = (anchor(view) for view in views)
                        prediction, a, b = model(raw, mask, views)
                        ce, gram, matches = dense_objective(
                            a, b, aa, ab, ia, ib, temperature=config["dense"]["temperature"]
                        )
                    else:
                        prediction = model(raw, mask)
                    loss, targets = prediction_loss(prediction, full, mask)
                    total = loss
                    if anchor is not None:
                        total = total + config["dense"]["correspondence_weight"] * ce
                        total = total + config["dense"]["gram_weight"] * gram
                    (total * len(batch_keys) / len(mine)).backward()
                weight = len(batch_keys) / len(mine)
                metrics_sum["prediction_loss"] += float(loss.detach()) * weight
                metrics_sum["total_loss"] += float(total.detach()) * weight
                if anchor is not None:
                    metrics_sum["dense_correspondence_loss"] += float(ce.detach()) * weight
                    metrics_sum["gram_anchor_loss"] += float(gram.detach()) * weight
                    metrics_sum["dense_exact_fraction"] += float(matches) * weight
                if step == 0 or (step + 1) % settings["log_interval"] == 0:
                    for name, value in representation_metrics(prediction, targets, full).items():
                        metrics_sum[name] += float(value) * weight
                metrics_sum["masked_fraction"] += float(mask.float().mean()) * weight
            grad_norm = torch.nn.utils.clip_grad_norm_(
                params, settings["gradient_clip"], error_if_nonfinite=True
            )
            rates = {f"optimization/lr_{g['name']}": g["lr"] for g in optimizer.param_groups}
            optimizer.step()
            scheduler.step()
            step += 1
            cursor = index + 1
            momentum = settings["ema_start"] + (settings["ema_end"] - settings["ema_start"]) * (
                step / settings["updates"]
            )
            update_target(target, online.encoder, momentum)
            stop = torch.tensor(int(paused), device=device)
            if world > 1:
                dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            paused = bool(stop.item())
            if step == 1 or step % settings["log_interval"] == 0:
                values = torch.tensor(list(metrics_sum.values()), device=device)
                if world > 1:
                    dist.all_reduce(values)
                metrics = {
                    f"train/{name}": float(value) / world
                    for name, value in zip(metrics_sum, values, strict=True)
                }
                metrics.update(rates)
                metrics.update(
                    {
                        "optimization/grad_norm": float(grad_norm),
                        "optimization/ema_momentum": momentum,
                        "progress/optimizer_updates": step,
                        "progress/epoch": epoch + cursor / len(batches),
                        "progress/windows_seen": step * settings["effective_batch"],
                        "progress/update_seconds": time.monotonic() - started,
                    }
                )
                if rank == 0:
                    emit(output, step, metrics)
                    write_json(
                        output / "status.json",
                        {
                            "status": "running",
                            "step": step,
                            "target": settings["updates"],
                            "prediction_loss": metrics["train/prediction_loss"],
                            "updated_at": time.time(),
                        },
                    )
            due = step % settings["validation_interval"] == 0 or step == settings["updates"]
            if due:
                if rank == 0:
                    write_json(
                        output / "status.json",
                        {"status": "validating", "step": step, "target": settings["updates"]},
                    )
                metrics = validate(
                    online, target, dataset, config, device, rank, world, step, anchor
                )
                if rank == 0:
                    emit(output, step, metrics)
            if (
                step == 1
                or step % settings["checkpoint_interval"] == 0
                or paused
                or step == settings["updates"]
            ):
                save()
            if paused or step == settings["updates"]:
                break
        if paused or step == settings["updates"]:
            break
        epoch += 1
        cursor = 0
    if rank == 0:
        write_json(
            output / "status.json",
            {
                "status": "paused" if paused else "completed",
                "step": step,
                "target": settings["updates"],
            },
        )
    if world > 1:
        dist.destroy_process_group()
