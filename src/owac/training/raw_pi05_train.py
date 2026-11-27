"""Bounded distributed RAW adaptation, with resumable state and offline selection."""

import argparse
import gc
import json
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from owac.raw.all_bench_distillation import load_inventory
from owac.training.j5_action_data import RawActionFrames, collate_actions
from owac.training.j5_action_metrics import action_metrics, action_sums
from owac.training.j5_action_train import Indices, disk_save, ema_weights, move
from owac.training.j5_lora import parameter_groups, trainable
from owac.training.j5_pi05 import observation
from owac.training.raw_pi05_model import adaptation_state, build_model
from owac.training.raw_pi05_schedule import BatchIndices, batch_partition, learning_rate_factor


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(),
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"])


def save(model, optimizer, ema, cfg, root, step, selection, rank, best=False):
    states = [None] * dist.get_world_size() if rank == 0 else None
    dist.gather_object(rng_state(), states, dst=0)
    if rank == 0:
        online = adaptation_state(model)
        averaged = dict(online)
        averaged.update({name: value.cpu() for name, value in ema.items()})
        payload = {
            "step": step,
            "config": cfg,
            "model": online,
            "ema": averaged,
            "optimizer": optimizer.state_dict(),
            "rng": states,
            "selection": selection,
        }
        disk_save(payload, root / "saving.pt")
        (root / "saving.pt").replace(root / "latest.pt")
        if best:
            (root / "best.pt").unlink(missing_ok=True)
            os.link(root / "latest.pt", root / "best.pt")
        (root / "selection.json").write_text(json.dumps(selection, indent=2))
    dist.barrier()


@torch.inference_mode()
def validate(model, loader, dataset, rank, world, device):
    model.eval()
    lights = ["Normal", "Low", "ExtremeLow", "Over", "ExtremeOver"]
    sums = torch.zeros((5, 8), device=device, dtype=torch.float64)
    offset = 0
    with torch.random.fork_rng(devices=[device.index]):
        for cpu_batch in loader:
            batch = move(cpu_batch, device)
            count = len(batch["raw"])
            noise = torch.stack(
                [
                    torch.randn(
                        (10, 32),
                        device=device,
                        generator=torch.Generator(device=device).manual_seed(
                            70000 + rank + i * world
                        ),
                    )
                    for i in range(offset, offset + count)
                ]
            )
            offset += count
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model.sample_actions(
                    device, observation(batch), noise=noise, num_steps=10
                )
            for light_index, light in enumerate(lights):
                chosen = torch.tensor([v == light for v in batch["lighting"]], device=device)
                if chosen.any():
                    sums[light_index] += action_sums(
                        prediction[chosen],
                        batch["actions"][chosen],
                        batch["action_valid"][chosen],
                        dataset.stats["actions"],
                        5,
                    )
    dist.all_reduce(sums)
    values = action_metrics(sums.sum(0).cpu().tolist())
    values["validation/anchors"] = len(dataset)
    for light, row in zip(lights, sums.cpu().tolist(), strict=True):
        values[f"validation/{light}_action_mse"] = row[0] / (7 * row[6])
    model.train()
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    train, root = cfg["training"], args.output
    root.mkdir(parents=True, exist_ok=True)
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    torch.set_num_threads(2)
    random.seed(cfg["seed"] + rank)
    np.random.seed(cfg["seed"] + rank)
    torch.manual_seed(cfg["seed"] + rank)
    latest = root / "latest.pt"
    model = build_model(cfg, latest if latest.exists() else None).to(device)
    parameters = trainable(model)
    optimizer = torch.optim.AdamW(
        parameter_groups(model, train),
        lr=train["learning_rate"],
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )
    start = 0
    selection = {"metric": "validation/action_mse", "best": None, "step": None}
    ema = {name: p.detach().clone() for name, p in parameters.items()}
    resume_rng = None
    previous_world = world
    if latest.exists():
        payload = torch.load(latest, map_location="cpu", weights_only=False, mmap=True)
        model.load_state_dict(payload["model"], strict=False)
        optimizer.load_state_dict(payload["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value) and value.device.type == "cpu":
                    state[key] = value.clone()
        ema = {name: payload["ema"][name].to(device).clone() for name in parameters}
        start, selection = payload["step"], payload["selection"]
        previous_world = len(payload["rng"])
        if rank < previous_world:
            resume_rng = payload["rng"][rank]
        if rank == 0:
            (root / f"resume-{start}.json").write_text(
                json.dumps(
                    {
                        "step": start,
                        "previous_world": previous_world,
                        "world": world,
                        "model_exact": all(
                            torch.equal(model.state_dict()[key].cpu(), value)
                            for key, value in payload["model"].items()
                        ),
                        "ema_exact": all(
                            torch.equal(ema[key].cpu(), payload["ema"][key]) for key in parameters
                        ),
                        "optimizer_entries": len(optimizer.state),
                        "optimizer_steps": sorted(
                            {int(value["step"]) for value in optimizer.state.values()}
                        ),
                        "restored_rng_ranks": list(range(min(previous_world, world))),
                        "new_rng_seeds": {
                            str(i): cfg["seed"] + start + i for i in range(previous_world, world)
                        },
                        "next_global_sample": start * train["batch"],
                        "global_batch": train["batch"],
                    },
                    indent=2,
                )
            )
        del payload
    ddp = DistributedDataParallel(
        model,
        device_ids=[device.index],
        broadcast_buffers=False,
        find_unused_parameters=cfg["adaptation"]["ranks"][0] > 0,
        gradient_as_bucket_view=True,
    )
    sources, _ = load_inventory(cfg["data"]["inventory"])
    sources = [s for s in sources if s.domain == "libero"]
    dataset = RawActionFrames([s for s in sources if s.split == "train"], cfg)
    validation = RawActionFrames(
        [s for s in sources if s.split == "validation"], cfg, validation=True
    )
    loader_kwargs = {
        "prefetch_factor": 1,
        "multiprocessing_context": "spawn",
        "pin_memory": True,
        "persistent_workers": True,
        "collate_fn": collate_actions,
    }
    uneven = train.get("allow_uneven_batch", False)
    local_batch = (
        batch_partition(train["batch"], world, rank)[1] if uneven else train["micro_batch"]
    )
    if not uneven and train["batch"] != world * local_batch:
        raise ValueError("Configured global batch must equal world size times micro batch")
    sampler = (
        BatchIndices(start * train["batch"], len(dataset), train["batch"], rank, world)
        if uneven
        else Indices(start * train["batch"], len(dataset), rank, world)
    )
    loader = DataLoader(
        dataset,
        batch_size=local_batch,
        num_workers=train["workers"],
        sampler=sampler,
        generator=torch.Generator().manual_seed(cfg["seed"] + rank),
        **loader_kwargs,
    )
    val_loader = DataLoader(
        validation,
        batch_size=cfg["validation"]["batch"],
        num_workers=1,
        sampler=Indices(0, len(validation), rank, world),
        generator=torch.Generator().manual_seed(70000 + rank),
        **loader_kwargs,
    )
    if rank == 0:
        (root / "config.json").write_text(json.dumps(cfg, indent=2))
        (root / "model.json").write_text(
            json.dumps(
                {
                    "trainable_parameters": sum(p.numel() for p in parameters.values()),
                    "trainable_names": list(parameters),
                    "source_counts": {
                        split: sum(s.split == split for s in sources)
                        for split in ("train", "validation", "test")
                    },
                    "validation_anchors": len(validation),
                    "global_batch": train["batch"],
                    "rank_batches": [
                        batch_partition(train["batch"], world, i)[1] for i in range(world)
                    ]
                    if uneven
                    else [local_batch] * world,
                },
                indent=2,
            )
        )

    def report(step, values):
        if rank == 0:
            record = {"step": step, "unix_time": time.time(), **values}
            with (root / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            (root / "status.json").write_text(json.dumps(record))
            print(json.dumps(record), flush=True)

    stop = [False]
    signal.signal(signal.SIGUSR1, lambda *_: stop.__setitem__(0, True))
    gc.collect()
    iterator = iter(loader)
    if resume_rng:
        restore_rng(resume_rng)
    else:
        seed = cfg["seed"] + start + rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    if selection["best"] is None:
        with ema_weights(model, ema, sharded=False):
            metrics = validate(model, val_loader, validation, rank, world, device)
        selection.update(best=metrics[selection["metric"]], step=start)
        report(start, metrics)
        save(model, optimizer, ema, cfg, root, start, selection, rank, best=True)
    model.train()
    previous = time.monotonic()
    for step in range(start + 1, train["updates"] + 1):
        batch = move(next(iterator), device)
        factor = learning_rate_factor(step, train)
        for group in optimizer.param_groups:
            group["lr"] = train["learning_rate"] * group["lr_scale"] * factor
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # DDP averages rank gradients; weight unequal local means by sample count.
            loss = ddp(batch).mean() * (world * local_batch / train["batch"])
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            list(parameters.values()), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        with torch.no_grad():
            for name, p in parameters.items():
                ema[name].lerp_(p.detach(), 1 - train["ema_decay"])
        elapsed = time.monotonic() - previous
        previous = time.monotonic()
        if step == start + 1 or step % train["log_every"] == 0:
            values = torch.stack([loss.detach().float(), norm.detach().float()])
            dist.all_reduce(values)
            report(
                step,
                {
                    "loss": values[0].item() / world,
                    "gradient_norm": values[1].item() / world,
                    "step_seconds": elapsed,
                    "samples_per_second": train["batch"] / elapsed,
                    "cuda_allocated_gib": torch.cuda.memory_allocated() / 2**30,
                    "cuda_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                },
            )
        request_stop = torch.tensor(int(stop[0]), device=device)
        dist.all_reduce(request_stop, op=dist.ReduceOp.MAX)
        paused = bool(request_stop.item()) or (
            args.stop_after is not None and step >= args.stop_after
        )
        final = step == train["updates"]
        best = False
        if step % train["eval_every"] == 0 or final:
            with ema_weights(model, ema, sharded=False):
                metrics = validate(model, val_loader, validation, rank, world, device)
            best = metrics[selection["metric"]] < selection["best"]
            if best:
                selection.update(best=metrics[selection["metric"]], step=step)
            report(step, metrics)
        if step == start + 1 or step % train["save_every"] == 0 or paused or final:
            save(model, optimizer, ema, cfg, root, step, selection, rank, best=best)
        if paused or final:
            if rank == 0:
                receipt = {"step": step, "native_exit": 0, "selection": selection}
                (root / ("complete.json" if final else "paused.json")).write_text(
                    json.dumps(receipt)
                )
            break
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
