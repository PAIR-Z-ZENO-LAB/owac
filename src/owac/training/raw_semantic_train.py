"""End-to-end RAW encoder learning from frozen paired-RGB DINOv3 dense semantics."""

import argparse
import json
import os
import random
import signal
import time
from collections import OrderedDict, defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from owac.raw.all_bench_distillation import (
    _restore_rng,
    _rng_state,
    load_inventory,
    read_npz_member,
)
from owac.training.raw_jepa_train import (
    RawWindows,
    emit,
    epoch_batches,
    lr_factor,
    save_file,
    write_json,
)
from owac.training.raw_semantic import build_semantic, semantic_loss


def detach_cpu_optimizer_state(optimizer):
    """Small Adam CPU counters must not retain the whole mmap checkpoint storage."""
    for state in optimizer.state.values():
        for name, value in state.items():
            if isinstance(value, torch.Tensor) and value.device.type == "cpu":
                state[name] = value.clone()


class PairedWindows(RawWindows):
    def __init__(self, sources, config):
        super().__init__(sources, config)
        self.rgb_cache = OrderedDict()
        self.capacity = config["training"]["working_cache_sources"]
        self.augmentation_group = config["training"].get("augmentation_group_windows", 2)

    def rgb(self, sid, endpoint):
        if sid not in self.rgb_cache:
            source = self.sources[sid]
            path = Path(str(source.path).replace("/libero/raw/", "/libero/rgb/"))
            self.rgb_cache[sid] = [
                read_npz_member(path, key + ".npy")
                for key in ("agentview_rgb_uint8", "wrist_rgb_uint8")
            ]
            if len(self.rgb_cache) > self.capacity:
                self.rgb_cache.popitem(last=False)
        self.rgb_cache.move_to_end(sid)
        return (
            torch.stack(
                [
                    torch.from_numpy(np.array(x[endpoint], copy=True)).permute(2, 0, 1)
                    for x in self.rgb_cache[sid]
                ]
            ).float()
            / 255
        )

    def paired(self, keys, device, *, augment):
        raw = self.native_batch(keys, device)
        rgb = torch.stack([self.rgb(s, e) for s, e in keys]).flatten(0, 1).to(device)
        # Keep the original two-window augmentation groups independent of GPU microbatch.
        group = 2 * self.augmentation_group if augment else len(raw)
        raws, rgbs = [], []
        for offset in range(0, len(raw), group):
            a, b = raw[offset : offset + group], rgb[offset : offset + group]
            if augment:
                size = random.choice((196, 224, 256))
                y, x = random.randrange(257 - size), random.randrange(257 - size)
                a, b = a[..., y : y + size, x : x + size], b[..., y : y + size, x : x + size]
                if random.random() < 0.5:
                    a, b = a.flip(-1), b.flip(-1)
            raws.append(
                F.interpolate(
                    a.flatten(0, 1), size=(224, 224), mode="bilinear", align_corners=False
                ).reshape(-1, 6, 3, 224, 224)
            )
            rgbs.append(F.interpolate(b, size=(256, 256), mode="bilinear", align_corners=False))
        return torch.cat(raws), torch.cat(rgbs)


class DinoTargets:
    def __init__(self, path, device):
        from transformers import AutoModel

        self.model = (
            AutoModel.from_pretrained(path, local_files_only=True)
            .requires_grad_(False)
            .eval()
            .to(device)
        )
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]

    @torch.no_grad()
    def __call__(self, rgb):
        output = self.model(pixel_values=(rgb - self.mean) / self.std).last_hidden_state
        return output[:, 1 + self.model.config.num_register_tokens :]


@torch.no_grad()
def validate(model, teacher, dataset, config, device, rank, world):
    model.eval()
    rows = []
    keys = dataset.validation_keys()
    for index in range(rank, len(keys), world):
        key = keys[index]
        raw, rgb = dataset.paired([key], device, augment=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            features = model(raw)
            target = teacher(rgb)
        loss, values = semantic_loss(features, target, **config["loss"])
        values = {k: float(v) for k, v in values.items()}
        values["total_loss"] = float(loss)
        values["patch_std"] = float(
            F.normalize(features.float(), dim=-1).std(1, correction=0).mean()
        )
        rows.append((dataset.sources[key[0]].lighting, values))
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, rows)
        rows = [r for items in gathered for r in items]
    groups = defaultdict(list)
    for light, values in rows:
        groups[light].append(values)
        groups["all"].append(values)
    metrics = {}
    for name, values in groups.items():
        metrics[f"validation/{name}/windows"] = len(values)
        for key in values[0]:
            metrics[f"validation/{name}/{key}"] = sum(row[key] for row in values) / len(values)
    model.train()
    return metrics


def train(config, output, resume=None):
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    torch.set_num_threads(config["training"].get("cpu_threads", 4))
    if world > 1:
        dist.init_process_group("nccl")
    cfg = config["training"]
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    sources, _ = load_inventory(config["data"]["inventory"])
    ds = PairedWindows(sources, config)
    model = build_semantic(config, initialize=resume is None).to(device)
    teacher = DinoTargets(config["semantic"]["teacher"], device)
    new_names = ("log_conv", "content_gate", "semantic_projection", "depth_logits")
    groups = [
        {
            "params": [p for n, p in model.named_parameters() if any(k in n for k in new_names)],
            "lr": cfg["head_lr"],
            "name": "new",
        },
        {
            "params": [
                p for n, p in model.named_parameters() if not any(k in n for k in new_names)
            ],
            "lr": cfg["backbone_lr"],
            "name": "inherited",
        },
    ]
    optimizer = torch.optim.AdamW(
        groups, betas=tuple(cfg["betas"]), weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_factor(s, cfg))
    step, epoch, cursor = 0, 0, 0
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=False, mmap=True)
        model.load_state_dict(state["encoder"])
        optimizer.load_state_dict(state["optimizer"])
        detach_cpu_optimizer_state(optimizer)
        scheduler.load_state_dict(state["scheduler"])
        step, epoch, cursor = state["step"], state["epoch"], state["cursor"]
        _restore_rng(state["rng_by_rank"][rank], local)
        del state
    else:
        torch.manual_seed(cfg["seed"] + rank)
        random.seed(cfg["seed"] + rank)
    parallel = DDP(model, device_ids=[local]) if world > 1 else model
    output.mkdir(parents=True, exist_ok=True)
    paused = False

    def pause(signum, frame):
        nonlocal paused
        paused = True

    signal.signal(signal.SIGUSR1, pause)

    def save():
        states = [None] * world
        if world > 1:
            dist.all_gather_object(states, _rng_state(local))
        else:
            states[0] = _rng_state(local)
        if rank == 0:
            payload = {
                "encoder": model.state_dict(),
                "config": config,
                "step": step,
                "schema": "owac.raw-semantic.encoder.v1",
            }
            write_json(output / "checkpoint.json", {"complete": False, "step": step})
            save_file(output / "encoder.pt", payload, output, config["storage"])
            payload.update(
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                epoch=epoch,
                cursor=cursor,
                rng_by_rank=states,
            )
            save_file(output / "latest.pt", payload, output, config["storage"])
            write_json(
                output / "checkpoint.json",
                {"complete": True, "encoder_complete": True, "step": step},
            )
        if world > 1:
            dist.barrier()

    if rank == 0:
        write_json(output / "config.json", config)
        write_json(
            output / "data-summary.json",
            {
                "sources": len(ds.sources),
                "train_windows": len(ds.keys("train")),
                "validation_windows": len(ds.validation_keys()),
                "trainable_parameters": sum(p.numel() for p in model.parameters()),
                "teacher_parameters": sum(p.numel() for p in teacher.model.parameters()),
                "RGB_teacher_training_only": True,
                "optimizer_updates_target": cfg["updates"],
            },
        )
        write_json(
            output / "status.json", {"status": "validating", "step": step, "target": cfg["updates"]}
        )
    if resume is None:
        metrics = validate(model, teacher, ds, config, device, rank, world)
        if rank == 0:
            emit(output, step, metrics)
    keys = ds.keys("train")
    while step < cfg["updates"]:
        batches = epoch_batches(keys, cfg["effective_batch"], cfg["seed"], epoch)
        for index in range(cursor, len(batches)):
            started = time.monotonic()
            mine = batches[index][rank::world]
            optimizer.zero_grad(set_to_none=True)
            totals = defaultdict(float)
            micro = cfg["micro_batch_per_rank"]
            for offset in range(0, len(mine), micro):
                batch = mine[offset : offset + micro]
                raw, rgb = ds.paired(batch, device, augment=True)
                sync = (
                    nullcontext()
                    if world == 1 or offset + micro >= len(mine)
                    else parallel.no_sync()
                )
                with sync, torch.autocast("cuda", dtype=torch.bfloat16):
                    target = teacher(rgb)
                    features = parallel(raw)
                    loss, values = semantic_loss(features, target, **config["loss"])
                    weight = len(batch) / len(mine)
                    (loss * weight).backward()
                for key, value in {"total_loss": loss, **values}.items():
                    totals[key] += value.detach() * weight
            grad = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            step += 1
            cursor = index + 1
            stop = torch.tensor(int(paused), device=device)
            if world > 1:
                dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            paused = bool(stop.item())
            if step == 1 or step % cfg["log_interval"] == 0:
                numbers = torch.stack(list(totals.values()))
                if world > 1:
                    dist.all_reduce(numbers)
                metrics = {
                    f"train/{key}": value / world
                    for key, value in zip(totals, numbers.cpu().tolist(), strict=True)
                }
                metrics.update(
                    {
                        "optimization/grad_norm": float(grad),
                        "progress/optimizer_updates": step,
                        "progress/epoch": epoch + cursor / len(batches),
                        "progress/windows_seen": step * cfg["effective_batch"],
                        "progress/update_seconds": time.monotonic() - started,
                    }
                )
                metrics.update(
                    {f"optimization/lr_{g['name']}": g["lr"] for g in optimizer.param_groups}
                )
                metrics.update(
                    {
                        f"architecture/depth{depth}_weight": float(w.detach())
                        for depth, w in zip(
                            model.depths, model.depth_logits.softmax(0), strict=True
                        )
                    }
                )
                if rank == 0:
                    emit(output, step, metrics)
                    write_json(
                        output / "status.json",
                        {
                            "status": "running",
                            "step": step,
                            "target": cfg["updates"],
                            "updated_at": time.time(),
                        },
                    )
            if step % cfg["validation_interval"] == 0 or step == cfg["updates"]:
                metrics = validate(model, teacher, ds, config, device, rank, world)
                if rank == 0:
                    emit(output, step, metrics)
            if (
                step == 1
                or step % cfg["checkpoint_interval"] == 0
                or paused
                or step == cfg["updates"]
            ):
                save()
            if paused or step == cfg["updates"]:
                break
        if paused or step == cfg["updates"]:
            break
        epoch += 1
        cursor = 0
    if rank == 0:
        write_json(
            output / "status.json",
            {"status": "paused" if paused else "completed", "step": step, "target": cfg["updates"]},
        )
    if world > 1:
        dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    if (
        args.resume
        and not json.loads((args.resume.parent / "checkpoint.json").read_text())["complete"]
    ):
        raise RuntimeError("Checkpoint write incomplete")
    train(json.loads(args.config.read_text()), args.output, args.resume)


if __name__ == "__main__":
    main()
