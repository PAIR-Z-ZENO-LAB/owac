"""Train and select an OWAC C1 Regression policy on physical fruit demonstrations."""

import argparse
import copy
import hashlib
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
from transformers import AutoTokenizer

from owac.frameworks.raw_compact.factory import make_policy
from owac.raw.all_bench_distillation import load_inventory
from owac.training.fruit_grasp import NeroHand, banana_validation, evaluate_approach, grasp_phase
from owac.training.fruit_placement import evaluate_placement, placement_events
from owac.training.raw_ssl_train import atomic_save, rng_state
from owac.training.raw_visual_initialization import initialize_visual
from owac.training.teleop_policy_data import FruitFrames, fit_statistics
from owac.training.teleop_policy_eval import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-tracking", action="store_true")
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    dist.init_process_group("nccl", device_id=device)
    torch.backends.cuda.matmul.allow_tf32 = True
    cfg = json.loads(args.config.read_text())
    sources, _ = load_inventory(cfg["data"]["inventory"])
    action_subsets = cfg["data"].get("action_subsets")
    train_src = [
        s
        for s in sources
        if s.split == "train" and (action_subsets is None or s.suite in action_subsets)
    ]
    valid_src = [s for s in sources if s.split == "validation"]
    if sorted(s.group_id for s in valid_src) != sorted(cfg["evaluation"]["validation_episodes"]):
        raise ValueError("Validation trajectory IDs differ from the fixed split")
    parent = cfg.get("continuation")
    if rank == 0:
        if parent and parent.get("preserve_statistics"):
            source = torch.load(
                parent["checkpoint"], map_location="cpu", mmap=True, weights_only=False
            )
            cfg["statistics"] = source["config"]["statistics"]
            del source
        else:
            cfg["statistics"] = fit_statistics(cfg, train_src)
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["assets"]["language_encoder"], local_files_only=True
        )
        text = tokenizer(
            cfg["task_names"][0],
            padding="max_length",
            max_length=cfg["model"]["language_length"],
            truncation=True,
        )
        cfg["task_token_ids"] = text["input_ids"]
        cfg["task_token_mask"] = list(map(bool, text["attention_mask"]))
    configs = [cfg]
    dist.broadcast_object_list(configs, src=0)
    cfg = configs[0]
    tc = cfg["training"]
    if tc["batch"] % world:
        raise ValueError("Global batch must divide evenly across ranks")
    torch.manual_seed(cfg["seed"])
    model = make_policy(cfg, initialize=args.resume is None and not parent).to(device)
    if parent and not args.resume:
        path = Path(parent["checkpoint"])
        with path.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != parent["sha256"]:
                raise ValueError("Fruit continuation checkpoint checksum differs")
        saved = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        if saved["step"] != parent["step"] or any(
            cfg[k] != saved["config"][k] for k in ("observation", "statistics", "task_names")
        ):
            raise ValueError(
                "Fruit continuation requires identical action and observation coordinates"
            )
        weights = saved["model"]
        projection = "resampler.position_projection.weight"
        if cfg["model"].get("spatial_coordinates") and projection not in weights:
            weights[projection] = model.state_dict()[projection]
        if cfg["model"].get("banana_grounding") and not saved["config"]["model"].get(
            "banana_grounding"
        ):
            for name, parameter in model.state_dict().items():
                if name.startswith(("spatial_goal_head.", "banana_projection.")):
                    weights[name] = parameter
        model.load_state_dict(weights, strict=True)
        if "new_visual_step" in parent:
            visual = initialize_visual(model.raw_encoder, cfg)
            if visual["visual_source_step"] != parent["new_visual_step"]:
                raise ValueError("Visual initialization step differs from the continuation recipe")
        if cfg["assets"].get("banana_head"):
            localizer = torch.load(
                cfg["assets"]["banana_head"], map_location="cpu", weights_only=False
            )
            if sorted(localizer["settings"]["training_episodes"]) != sorted(
                s.group_id for s in train_src
            ):
                raise ValueError("The banana locator was fitted on a different training split")
            model.spatial_goal_head.load_state_dict(localizer["head"], strict=True)
        model.initialization = parent
        del saved
    vision = list(model.raw_encoder.parameters())
    rest = [v for k, v in model.named_parameters() if not k.startswith("raw_encoder.")]
    opt = torch.optim.AdamW(
        [dict(params=rest), dict(params=vision)],
        lr=tc["learning_rate"],
        weight_decay=tc["weight_decay"],
        betas=(0.9, 0.95),
        fused=True,
    )
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    start, best, best_step = 0, math.inf, None
    best_dark = math.inf
    best_grasp = math.inf
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", mmap=True, weights_only=False)
        if saved["config"] != cfg or len(saved["rng"]) != world:
            raise ValueError("Resume requires the same fruit recipe and rank count")
        model.load_state_dict(saved["model"])
        ema.load_state_dict(saved["ema"])
        opt.load_state_dict(saved["optimizer"])
        start, best, best_step = saved["step"], saved["best"], saved["best_step"]
        best_dark = saved.get("best_dark", math.inf)
        best_grasp = saved.get("best_grasp", math.inf)
        state = saved["rng"][rank]
        torch.set_rng_state(state["torch"])
        torch.cuda.set_rng_state(state["cuda"])
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        del saved
    else:
        torch.manual_seed(cfg["seed"] + rank)
    net = DDP(
        model, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=True
    )
    if rank == 0:
        (root / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
    if rank == 0 and not args.no_tracking:
        import swanlab

        track = cfg["swanlab"]
        swanlab.login()
        run = swanlab.init(
            workspace=track["workspace"],
            project=track["project"],
            id=track["id"],
            name=track["name"],
            group=track["group"],
            job_type=track["job_type"],
            resume="allow",
            public=False,
            config=cfg,
            logdir=str(root / "swanlab"),
        )
        (root / "swanlab.json").write_text(json.dumps(dict(url=run.url)))

    def report(step, values):
        if rank == 0:
            with (root / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(dict(step=step, metrics=values), allow_nan=False) + "\n")
            if not args.no_tracking:
                swanlab.log(values, step=step)
            print(json.dumps(dict(step=step, metrics=values)), flush=True)

    def scores(method, prefix, exposure=0):
        values, per = evaluate(
            ema, val, method=method, linear=linear, rank=rank, world=world, exposure=exposure
        )
        if cfg["evaluation"].get("replay_selected") and prefix == "validation":
            for episode, row in per.items():
                for key in ("normalized_l1", "joint_mae_deg", "gripper_mae_mm"):
                    values[f"{episode}/{key}"] = row[key]
        return {prefix + "/" + k: v for k, v in values.items()}, per

    def export(name, step):
        atomic_save(dict(model=ema.state_dict(), step=step, config=cfg), root / name)

    train, val = FruitFrames(train_src, cfg, device), FruitFrames(valid_src, cfg, device)
    linear = train.linear_baseline()
    hand = None
    placements = (
        [placement_events(src) for src in valid_src] if cfg["evaluation"].get("placement") else []
    )
    if "grasp_evaluation" in cfg:
        hand = NeroHand(cfg["grasp_evaluation"]["urdf"], device)
        scales = hand.reference_scale(train)
        seconds = cfg["grasp_evaluation"]["seconds"]
        remaining = np.concatenate([grasp_phase(src, seconds)[0] for src in valid_src])

    def grasp_scores():
        values = evaluate_approach(ema, val, hand, scales, remaining, seconds)
        values = {"grasp/" + k: v for k, v in values.items()}
        if train.banana_mass is not None:
            values.update(banana_validation(ema, train, val))
        return values

    def placement_scores(step):
        values, episodes = evaluate_placement(ema, val, hand, placements)
        if rank == 0:
            (root / f"placement-{step:06d}.json").write_text(
                json.dumps(dict(step=step, metrics=values, episodes=episodes), indent=2)
            )
        return {"placement/" + k: v for k, v in values.items()}

    if rank == 0:
        (root / "data.json").write_text(
            json.dumps(
                dict(
                    train={s.group_id: int(n) for s, n in zip(train_src, train.sizes, strict=True)},
                    validation={
                        s.group_id: int(n) for s, n in zip(valid_src, val.sizes, strict=True)
                    },
                    train_commands=int(train.valid.sum()),
                    valid_commands=int(val.valid.sum()),
                    encoder_initialization=model.initialization,
                    parameters=sum(p.numel() for p in model.parameters()),
                ),
                indent=2,
            )
        )
    if not start:
        hold, _ = scores("hold", "baseline/hold")
        regression, _ = scores("linear", "baseline/state_linear")
        report(0, hold | regression)

    running = torch.zeros(6, device=device)
    count = 0
    stop = [False]
    signal.signal(signal.SIGUSR1, lambda *_: stop.__setitem__(0, True))
    step = start
    for step in range(start + 1, tc["updates"] + 1):
        frozen = step <= tc["frozen_steps"]
        mode = "features" if frozen else "raw"
        if train.mode != mode:
            train.load_views(mode)
            val.load_views(mode)
            if step == 1:
                zero, _ = scores("policy", "validation")
                if hand is not None:
                    zero.update(grasp_scores())
                if placements:
                    zero.update(placement_scores(0))
                if cfg["evaluation"].get("exposure_stops"):
                    dark, _ = scores(
                        "policy", "validation_dark", cfg["evaluation"]["exposure_stops"]
                    )
                    zero.update(dark)
                report(0, zero)
            dist.barrier()
            clock = time.monotonic()
        model.train()
        frac = max(0, step - tc["warmup"]) / (tc["updates"] - tc["warmup"])
        lr = (
            tc["learning_rate"]
            * min(1, step / tc["warmup"])
            * (0.1 + 0.9 * (1 + math.cos(math.pi * frac)) / 2)
        )
        opt.param_groups[0]["lr"] = lr
        opt.param_groups[1]["lr"] = (
            0 if frozen or model.freeze_raw_encoder else lr * tc["visual_lr_multiplier"]
        )
        ix = train.sample_indices(cfg["seed"] + step * world + rank, tc["batch"] // world)
        batch = train.training_batch(ix, cfg["seed"] + step * world + rank)
        native_size = len(ix)
        paired = tc.get("paired_low_light")
        if paired:
            ix, paired_rows = train.append_light_pairs(batch, ix, cfg["seed"] + step * world + rank)
        opt.zero_grad(set_to_none=True)
        pose_objective = tc.get("hand_pose_loss", False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = net(batch, return_prediction=True) if pose_objective or paired else net(batch)
            loss, mse, _ = result[:3]
        pose_loss = loss.detach().new_zeros(())
        if pose_objective:
            physical = train.physical(result[3].float(), ix)
            pose_loss = hand.loss(physical, train.absolute[ix], train.valid[ix], scales)
            loss = (loss + pose_loss) / 2
            if train.banana_mass is not None:
                loss = (2 * loss + result[4] * scales[2]) / 3
        consistency = loss.detach().new_zeros(())
        if paired and len(paired_rows):
            prediction = result[3].float()
            delta = prediction[native_size:] - prediction[paired_rows].detach()
            consistency = delta[batch["action_valid"][native_size:]].square().mean()
            loss = loss + paired["weight"] * consistency
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        with torch.no_grad():
            torch._foreach_lerp_(
                list(ema.parameters()), list(model.parameters()), 1 - tc["ema_decay"]
            )
        heatmap = (
            result[4].detach() if train.banana_mass is not None else loss.detach().new_zeros(())
        )
        running += torch.stack(
            (
                mse.detach(),
                norm.detach(),
                pose_loss.detach(),
                loss.detach(),
                heatmap,
                consistency.detach(),
            )
        )
        count += 1
        if step % tc["log_every"] == 0 or step == start + 1:
            dist.all_reduce(running)
            mean = (running / (count * world)).tolist()
            report(
                step,
                {
                    "train/regression_mse": mean[0],
                    "train/grad_norm": mean[1],
                    "train/lr": lr,
                    "train/visual_lr": opt.param_groups[1]["lr"],
                    **({"train/light_consistency_mse": mean[5]} if paired else {}),
                    **(
                        {"train/banana_heatmap_loss": mean[4]}
                        if train.banana_mass is not None
                        else {}
                    ),
                    **(
                        {"train/pose_loss": mean[2], "train/objective": mean[3]}
                        if pose_objective
                        else {}
                    ),
                    "runtime/seconds_per_update": (time.monotonic() - clock) / count,
                },
            )
            running.zero_()
            count = 0
            clock = time.monotonic()
            signal_value = torch.tensor(int(stop[0]), device=device)
            dist.all_reduce(signal_value, op=dist.ReduceOp.MAX)
            stop[0] = bool(signal_value.item())
            if rank == 0:
                (root / "status.json").write_text(
                    json.dumps(dict(step=step, status="running", stage=mode))
                )
        ending = step == tc["updates"] or stop[0]
        if step % tc["eval_every"] == 0 or ending:
            values, per = scores("policy", "validation")
            if hand is not None:
                grasp = grasp_scores()
                grasp_score = grasp["grasp/normalized_error"]
                if grasp_score < best_grasp:
                    best_grasp = grasp_score
                    if rank == 0:
                        export("policy-grasp.pt", step)
                        (root / "grasp.json").write_text(
                            json.dumps(dict(step=step, metrics=grasp), indent=2)
                        )
                values.update(grasp)
            if placements:
                values.update(placement_scores(step))
            score = values[cfg["evaluation"]["metric"]]
            if score < best:
                best, best_step = score, step
                if rank == 0:
                    export("policy-best.pt", step)
                    (root / "best.json").write_text(
                        json.dumps(dict(step=step, metrics=values, episodes=per), indent=2)
                    )
            if cfg["evaluation"].get("exposure_stops"):
                dark, episodes = scores(
                    "policy", "validation_dark", cfg["evaluation"]["exposure_stops"]
                )
                quality = dark["validation_dark/macro_normalized_mse"]
                if quality < best_dark:
                    best_dark = quality
                    if rank == 0:
                        export("policy-dark.pt", step)
                        (root / "dark.json").write_text(
                            json.dumps(dict(step=step, metrics=dark, episodes=episodes), indent=2)
                        )
                values.update(dark)
            if step in (tc["frozen_steps"], tc["updates"]):
                wrong, _ = scores("wrong_views", "diagnostic/wrong_views")
                values.update(wrong)
                if rank == 0:
                    export("policy-frozen.pt" if frozen else "policy-final.pt", step)
            report(step, values)
            clock = time.monotonic()
        if step % tc["save_every"] == 0 or ending:
            states = [None] * world
            dist.all_gather_object(states, rng_state())
            if rank == 0:
                atomic_save(
                    dict(
                        model=model.state_dict(),
                        ema=ema.state_dict(),
                        optimizer=opt.state_dict(),
                        rng=states,
                        config=cfg,
                        step=step,
                        best=best,
                        best_step=best_step,
                        best_dark=best_dark,
                        best_grasp=best_grasp,
                    ),
                    root / "latest.pt",
                )
            dist.barrier()
            clock = time.monotonic()
        if ending:
            break
    if cfg["evaluation"].get("replay_selected") and step == tc["updates"]:
        dist.barrier()
        selected = torch.load(
            root / "policy-best.pt", map_location="cpu", mmap=True, weights_only=False
        )
        ema.load_state_dict(selected["model"], strict=True)
        val.load_views("raw")
        predictions = np.zeros(tuple(val.absolute.shape), dtype=np.float32)
        metrics, per = evaluate(ema, val, rank=rank, world=world, predictions=predictions)
        if world > 1:
            gathered = torch.from_numpy(predictions).to(device)
            dist.reduce(gathered, dst=0)
            if rank == 0:
                predictions = gathered.cpu().numpy()
        replay = {
            f"replay/{key}": float(np.mean([row[key] for row in per.values()]))
            for key in ("normalized_l1", "joint_mae_deg", "gripper_mae_mm")
        }
        replay.update({f"replay/{ep}/{k}": v for ep, row in per.items() for k, v in row.items()})
        report(step, replay | {"replay/selected_step": selected["step"]})
        if rank == 0:
            replay_dir = root / "replay"
            replay_dir.mkdir(exist_ok=True)
            for source, start_row, size in zip(val.sources, val.starts, val.sizes, strict=True):
                rows = slice(start_row, start_row + size)
                np.savez_compressed(
                    replay_dir / (source.group_id.replace("/", "-") + ".npz"),
                    predicted=predictions[rows],
                    target=val.absolute[rows].cpu().numpy(),
                    state=val.state[rows].cpu().numpy(),
                    valid=val.valid[rows].cpu().numpy(),
                )
            (replay_dir / "results.json").write_text(
                json.dumps(
                    dict(method="owac", step=selected["step"], validation=per, metrics=metrics),
                    indent=2,
                    allow_nan=False,
                )
            )
    if rank == 0:
        (root / "status.json").write_text(
            json.dumps(
                dict(
                    step=step,
                    status="paused" if stop[0] else "completed",
                    best_step=best_step,
                    best=best,
                )
            )
        )
        if not args.no_tracking:
            swanlab.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
