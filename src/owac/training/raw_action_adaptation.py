"""R1: native demonstration flow matching through a frozen LIBERO π0.5 backend."""

from __future__ import annotations

import json
import math
import os
import random
import signal
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from owac.raw.all_bench_distillation import (
    AllBenchDataset,
    EpisodeCache,
    _chunks_for_epoch,
    _restore_rng,
    _rng_state,
    build_all_bench_student,
    load_inventory,
)
from owac.training.raw_full_visual_batched import BatchedTemporalRequest


def normalize_window(actions, states, endpoint, stats):
    """Native LIBERO quantiles, action[t:t+10], with terminal padding masked."""
    if actions.shape != (len(states), 7) or states.shape[1:] != (8,):
        raise ValueError("LIBERO requires aligned actions[T,7] and robot_states[T,8]")
    ids = np.arange(endpoint, endpoint + 10)
    valid = ids < len(actions)
    values = actions[np.minimum(ids, len(actions) - 1)].copy()
    state = states[endpoint].copy()
    for value, key in ((values, "actions"), (state, "state")):
        width = value.shape[-1]
        lo = np.asarray(stats[key]["q01"], dtype=np.float32)[:width]
        hi = np.asarray(stats[key]["q99"], dtype=np.float32)[:width]
        value[:] = np.clip(2 * (value - lo) / np.maximum(hi - lo, 1e-6) - 1, -1, 1)
    padded = np.zeros((10, 32), np.float32)
    padded[:, :7] = values
    return state, padded, valid


class ActionDataset(AllBenchDataset):
    """Reuse causal RAW packet construction, without constructing a target store."""

    def __init__(self, sources, stats, capacity=16):
        self.sources = {s.group_id: s for s in sources if s.domain == "libero"}
        self.cache = EpisodeCache(self.sources, capacity=capacity)
        self.stats = stats
        self.labels = {}

    def target(self, source_id, endpoint):
        if source_id not in self.labels:
            path = self.sources[source_id].path
            # These small labels are owned arrays, so no per-source mmap handles remain.
            with np.load(path, allow_pickle=False) as archive:
                self.labels[source_id] = (
                    archive["actions"],
                    archive["robot_states"],
                    json.loads(str(archive["metadata_json"].item()))["task_description"],
                )
        actions, states, prompt = self.labels[source_id]
        state, action, valid = normalize_window(actions, states, endpoint, self.stats)
        return SimpleNamespace(state=state, actions=action, valid=valid, prompt=prompt)

    def sample(self, source_id, endpoint):
        unroll = tuple(endpoint - 25 + 5 * i for i in range(6))
        prefix = tuple(range(0, unroll[0], 5))
        frames = self._frame_cache(source_id, endpoint)
        return SimpleNamespace(
            source_id=source_id,
            endpoint=endpoint,
            unroll_packets=tuple(self._packet_from_cache(source_id, e, frames) for e in unroll),
            prefix_packets=tuple(self._packet_from_cache(source_id, e, frames) for e in prefix),
            target=self.target(source_id, endpoint),
        )

    def anchor_keys(self, split="train"):
        return [
            (s.group_id, e)
            for s in self.sources.values()
            if s.split == split
            for e in s.train_endpoints
        ]


def native_flow_loss(bridge, visual, visual_mask, targets, *, noise=None, flow_time=None):
    """Keep native prefix order, missing-camera mask, flow targets and action expert."""
    from owac.backends.rawvla_language import prepare_language_state

    base = bridge.base
    device = visual.device
    lang, lang_mask, state = prepare_language_state(
        bridge, [t.prompt for t in targets], np.stack([t.state for t in targets]), device=device
    )
    actions = torch.as_tensor(np.stack([t.actions for t in targets]), device=device)
    valid = torch.as_tensor(np.stack([t.valid for t in targets]), device=device)
    language = base.vlm.embed_language_tokens(lang)
    # Native third (right wrist) camera is absent in LIBERO. Preserve its masked slot.
    missing = visual.new_zeros((len(targets), 256, visual.shape[-1]))
    prefix = torch.cat(
        (visual.to(language), missing.to(language), language * math.sqrt(language.shape[-1])), dim=1
    )
    prefix_mask = torch.cat(
        (visual_mask, visual_mask.new_zeros((len(targets), 256)), lang_mask), dim=1
    )
    x_t, u_t, _, tau = base.action_head.prepare_flow_targets(actions, noise=noise, time=flow_time)
    suffix, suffix_pad, suffix_att, cond = base.action_head.embed_action_suffix(state, x_t, tau)
    pad = torch.cat((prefix_mask, suffix_pad), dim=1)
    att = torch.cat((torch.zeros_like(prefix_mask), suffix_att), dim=1)
    out = base.forward_with_action_expert(
        prefix_embs=prefix,
        suffix_embs=suffix.to(prefix.dtype),
        attention_mask=bridge.mask_4d(bridge.mask_2d(pad, att)),
        position_ids=pad.long().cumsum(1) - 1,
        adarms_cond=cond,
    )
    velocity = base.action_head.decode_action_velocity(out)
    squared = (velocity.float() - u_t.float()).square().mean(dim=-1)
    return (squared * valid).sum(), valid.sum()


def checkpoint_backend(base):
    """Recompute native shared layers while retaining input gradients through frozen weights."""
    original = base.forward_shared_gemma_layer

    def forward(index, models, streams, conds, attention_mask, position_ids):
        if not torch.is_grad_enabled():
            return original(index, models, streams, conds, attention_mask, position_ids)

        def layer(prefix, suffix):
            return original(index, models, [prefix, suffix], conds, attention_mask, position_ids)

        return checkpoint(layer, *streams, use_reentrant=False)

    base.forward_shared_gemma_layer = forward


def build_backend(config, device):
    import sys

    from owac.backends.rawvla_features_n02 import build_feature_policy, load_checkpoint_state

    assets = config["assets"]
    sys.path.insert(0, assets["starvla_root"])
    bridge = build_feature_policy(
        "libero",
        Path(assets["tokenizer"]),
        None,
        token_dim=2048,
        tokens_per_camera=256,
        image_keys=("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
        freeze_backend=True,
    )
    bridge.load_audited_replaced_vision(load_checkpoint_state(assets["checkpoint"]))
    bridge.release_replaced_vision()
    bridge.base.to(device).eval()
    checkpoint_backend(bridge.base)
    return bridge


class BudgetWriter:
    """Limit bytes during serialization, counting only the current small run directory."""

    def __init__(self, stream, allowance):
        self.stream, self.allowance = stream, allowance

    def write(self, data):
        if self.stream.tell() + len(data) > self.allowance:
            raise OSError("R1 checkpoint would exceed the inherited 380GiB disk budget")
        return self.stream.write(data)

    def flush(self):
        return self.stream.flush()


def save_budgeted(path, payload, output, baseline_bytes, cap_bytes):
    used = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as stream:
        torch.save(payload, BudgetWriter(stream, cap_bytes - baseline_bytes - used))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def append_json(path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(value) + "\n")


def validate(student, bridge, dataset, rank, world, device, step):
    from owac.runtime.raw_full_visual_distillation import set_full_visual_train_mode

    set_full_visual_train_mode(student, False)
    totals = {}
    sources = [s for s in dataset.sources.values() if s.split == "validation"]
    # Fixed flow draws make validation curves comparable, without perturbing training RNG.
    with torch.random.fork_rng(devices=[device.index]), torch.no_grad():
        for index, source in enumerate(sources):
            if index % world != rank:
                continue
            torch.manual_seed(1000000 + index)
            temporal = None
            for endpoint in source.endpoints:
                packet = dataset.packet_for_endpoint(source.group_id, endpoint, device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result, temporal = student(packet, temporal)
                    loss_sum, count = native_flow_loss(
                        bridge,
                        result.projected_tokens,
                        result.tokens.mask,
                        [dataset.target(source.group_id, endpoint)],
                    )
                for key in ("all", source.lighting, "startup" if endpoint < 25 else "eligible"):
                    row = totals.setdefault(key, [0.0, 0, 0])
                    row[0] += float(loss_sum)
                    row[1] += int(count)
                    row[2] += 1
    gathered = [None] * world
    if world > 1:
        dist.all_gather_object(gathered, totals)
    else:
        gathered[0] = totals
    merged = {}
    for part in gathered:
        for key, value in part.items():
            row = merged.setdefault(key, [0.0, 0, 0])
            for i in range(3):
                row[i] += value[i]
    set_full_visual_train_mode(student, True)
    return {
        "step": step,
        "groups": {
            key: {"flow_loss": row[0] / row[1], "valid_actions": row[1], "windows": row[2]}
            for key, row in merged.items()
        },
    }


def render_curves(output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train = [json.loads(line) for line in (output / "updates.jsonl").read_text().splitlines()]
    validation = [
        json.loads(line) for line in (output / "validation.jsonl").read_text().splitlines()
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot([r["step"] for r in train], [r["flow_loss"] for r in train], linewidth=0.6)
    axes[0].set_title("Training: native flow matching")
    for key in validation[-1]["groups"]:
        if key not in {"startup", "eligible"}:
            axes[1].plot(
                [r["step"] for r in validation],
                [r["groups"][key]["flow_loss"] for r in validation],
                label=key,
            )
    axes[1].set_title("Validation: fixed flow draws")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("Optimizer updates")
        ax.set_ylabel("Velocity MSE")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "curves.png", dpi=150)
    fig.savefig(output / "curves.svg")
    plt.close(fig)


def train(config, *, output, resume=None):
    from torch.nn.parallel import DistributedDataParallel as DDP

    from owac.runtime.raw_full_visual_distillation import (
        _checkpoint_wrap_vision,
        set_full_visual_train_mode,
    )

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if world > 1:
        dist.init_process_group("nccl")
    cfg = config["training"]
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    sources, _ = load_inventory(config["data"]["inventory"])
    stats = json.loads(Path(config["assets"]["statistics"]).read_text())["norm_stats"]
    dataset = ActionDataset(sources, stats, cfg["working_cache_sources"])
    keys = dataset.anchor_keys()
    student = build_all_bench_student(config, device=str(device))
    _checkpoint_wrap_vision(student)
    set_full_visual_train_mode(student, True)
    bridge = build_backend(config, device)
    model = DDP(student, device_ids=[local]) if world > 1 else student
    params = list(student.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / cfg["warmup_updates"])
    )
    output.mkdir(parents=True, exist_ok=True)
    step, epoch, cursor = 0, 0, 0
    if resume is not None:
        payload = torch.load(resume, map_location="cpu", weights_only=True)
        student.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        step, epoch, cursor = payload["global_step"], payload["epoch"], payload["cursor"]
        _restore_rng(payload["rng_by_rank"][rank], local)
        del payload
    else:
        torch.manual_seed(cfg["seed"] + rank)
    budget = config["storage"]
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
            save_budgeted(
                output / "latest.pt",
                {
                    "model": student.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "global_step": step,
                    "epoch": epoch,
                    "cursor": cursor,
                    "rng_by_rank": states,
                    "config": config,
                },
                output,
                budget["existing_owned_bytes"],
                budget["cap_bytes"],
            )
        if world > 1:
            dist.barrier()

    if rank == 0:
        write_json(output / "config.json", config)
        write_json(
            output / "data-summary.json",
            {
                "source_split": dict(Counter(s.split for s in dataset.sources.values())),
                "training_anchors": len(keys),
                "lighting": dict(Counter(s.lighting for s in dataset.sources.values())),
                "trainable_parameters": sum(p.numel() for p in params),
                "frozen_backend_parameters": sum(p.numel() for p in bridge.base.parameters()),
            },
        )
    while step < cfg["updates"]:
        chunks = _chunks_for_epoch(keys, dataset.sources, seed=cfg["seed"], epoch=epoch)
        for index in range(cursor, len(chunks)):
            started = time.monotonic()
            mine = chunks[index][rank::world]
            micro = cfg["micro_batch_per_rank"]
            # Weight the full global batch by real future action count, including episode tails.
            denominator = sum(int(dataset.target(s, e).valid.sum()) for s, e in chunks[index])
            optimizer.zero_grad(set_to_none=True)
            local_loss = torch.zeros((), device=device)
            for start in range(0, len(mine), micro):
                batch = [dataset.sample(s, e) for s, e in mine[start : start + micro]]
                sync = (
                    nullcontext() if world == 1 or start + micro >= len(mine) else model.no_sync()
                )
                with sync, torch.autocast("cuda", dtype=torch.bfloat16):
                    result = model(BatchedTemporalRequest(tuple(batch)))
                    loss_sum, _ = native_flow_loss(
                        bridge,
                        result.projected_tokens,
                        result.token_mask,
                        [b.target for b in batch],
                    )
                    loss = loss_sum * world / denominator
                    loss.backward()
                local_loss += loss_sum.detach()
                del result, loss, loss_sum, batch
            torch.nn.utils.clip_grad_norm_(params, cfg["gradient_clip"], error_if_nonfinite=True)
            lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            step += 1
            cursor = index + 1
            if world > 1:
                dist.all_reduce(local_loss)
            stop = torch.tensor(int(paused), device=device)
            if world > 1:
                dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            paused = bool(stop.item())
            if rank == 0:
                append_json(
                    output / "updates.jsonl",
                    {
                        "step": step,
                        "flow_loss": float(local_loss) / denominator,
                        "lr": lr,
                        "seconds": time.monotonic() - started,
                        "epoch": epoch,
                        "cursor": cursor,
                        "presentations": step * 16,
                    },
                )
                if step == 1 or step % 10 == 0:
                    write_json(
                        output / "status.json",
                        {
                            "status": "running",
                            "step": step,
                            "target": cfg["updates"],
                            "flow_loss": float(local_loss) / denominator,
                        },
                    )
            due = step % cfg["validation_interval_updates"] == 0 or step == cfg["updates"]
            if due:
                row = validate(student, bridge, dataset, rank, world, device, step)
                if rank == 0:
                    append_json(output / "validation.jsonl", row)
                    render_curves(output)
            if step == 1 or step % cfg["checkpoint_interval_updates"] == 0 or due or paused:
                save()
            if step == cfg["updates"] or paused:
                break
        if step == cfg["updates"] or paused:
            break
        epoch += 1
        cursor = 0
    if rank == 0:
        if not paused:
            save_budgeted(
                output / "final100000.pt",
                {
                    "model": student.state_dict(),
                    "global_step": step,
                    "config": config,
                },
                output,
                budget["existing_owned_bytes"],
                budget["cap_bytes"],
            )
        write_json(
            output / "status.json",
            {"status": "paused" if paused else "completed", "step": step, "target": cfg["updates"]},
        )
    if world > 1:
        dist.destroy_process_group()
