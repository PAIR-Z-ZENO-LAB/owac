"""Refit the existing OWAC action decoder on local bowl demonstrations and its own state errors."""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import swanlab
import torch

from owac.raw.all_bench_distillation import load_inventory
from owac.training.compact_train import atomic_save
from owac.training.fruit_grasp import NeroHand
from owac.training.teleop_recovery import fixed_language, load_export
from owac.training.teleop_recovery_replay import RecoveryBank, score_replay


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("parent", "experiment", "urdf", "run"):
        p.add_argument(arg, type=Path)
    p.add_argument("--id", required=True)
    p.add_argument("--updates", type=int, default=1600)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--send", type=int, default=5)
    p.add_argument("--batch", type=int, default=48)
    p.add_argument("--absolute", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    args.run.mkdir(exist_ok=True, parents=True)
    torch.manual_seed(20261124)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    model, parent_step = load_export(args.parent)
    cfg = model.cfg
    source, _ = load_inventory(args.experiment / "aligned/inventory.json")
    tracking = json.loads((args.experiment / "field-tracking.json").read_text())
    train, val = [
        RecoveryBank(
            [s for s in source if s.split == split], args.experiment / "visual", cfg, tracking
        )
        for split in ("train", "validation")
    ]
    text = fixed_language(model)
    for bank in (train, val):
        bank.configure_output(cfg)
    local_parent_errors = train.rollout(model, text, args.send)
    scales = train.half.clone()
    cpu_hand = NeroHand(args.urdf, "cpu")
    gpu_hand = NeroHand(args.urdf, "cuda")
    with torch.no_grad():
        base_results = {}
        for send in dict.fromkeys((12, args.send)):
            base_results[str(send)] = score_replay(val, val.rollout(model, text, send), cpu_hand)
        pos, rot = gpu_hand.squared_errors(train.q[:, None].expand_as(train.target), train.target)
        pose_scale = torch.stack((pos[train.valid].mean(), rot[train.valid].mean()))
    if args.absolute:
        cfg["observation"]["relative_action_mask"] = [False] * 8
        low, high = torch.quantile(
            train.target[train.valid], torch.tensor([0.01, 0.99], device="cuda"), dim=0
        )
        cfg["statistics"]["action_q01"] = low.tolist()
        cfg["statistics"]["action_q99"] = high.tolist()
    mode = "absolute" if args.absolute else "relative"
    cfg["recovery_training"] = dict(
        parent_step=parent_step,
        label_coordinates=mode,
        source_sha256=json.loads((args.experiment / "visual/source.json").read_text())["sha256"],
        train_episodes=train.names,
        validation_episodes=val.names,
        selection="count then mean 3D at two picks and two releases",
        training=(
            "Fixed vision and language, replayed state errors, feasible expert targets "
            "and gripper-frame pose loss"
        ),
        send=args.send,
        updates=args.updates,
        batch=args.batch,
        validation_kind="recorded images and reference-relative first-order follower replay",
    )
    for bank in (train, val):
        bank.configure_output(cfg)
    model.requires_grad_(False)
    params = []
    for block in (model.state_encoder, model.action_generator):
        block.requires_grad_(True)
        params.extend(block.parameters())
    optimizer = torch.optim.AdamW(params, lr=3e-5, betas=(0.9, 0.95), weight_decay=0.01, fused=True)
    sw = swanlab.init(
        workspace=os.environ.get("SWANLAB_WORKSPACE"),
        mode=os.environ.get("SWANLAB_MODE", "local"),
        project=os.environ.get("SWANLAB_PROJECT", "OWAC"),
        id=args.id,
        name=f"OWAC | bowl | local closed-loop replay | {mode} | {args.send} actions",
        public=False,
        resume="allow",
        group="real-bowl-20261123",
        job_type="action-policy",
        config=cfg["recovery_training"] | dict(gpu=torch.cuda.get_device_name()),
        logdir=str(args.run / "swanlog"),
    )
    (args.run / "swanlab.json").write_text(json.dumps(dict(url=sw.url)))
    (args.run / "config.json").write_text(json.dumps(cfg, indent=2))
    (args.run / "parent-replay.json").write_text(json.dumps(base_results, indent=2))

    def report(step, values):
        values = {k: v for k, v in values.items() if v is not None}
        with (args.run / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(dict(step=step, metrics=values), allow_nan=False) + "\n")
        swanlab.log(values, step=step)
        print(step, json.dumps(values), flush=True)

    start = 0
    best_score = (0, math.inf)
    current_errors = local_parent_errors
    if args.resume:
        saved = torch.load(
            args.run / "training-latest.pt", map_location="cpu", weights_only=False, mmap=True
        )
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start, best_score = saved["step"], tuple(saved["best_score"])
        current_errors = train.rollout(model, text, args.send)
    else:
        report(
            0,
            {
                f"original_{send}/{k}": v
                for send, (summary, _) in base_results.items()
                for k, v in summary.items()
            },
        )
    running = torch.zeros(3, device="cuda")
    updates = 0
    for step in range(start + 1, args.updates + 1):
        rng = np.random.default_rng(20261124 + step)
        ix = train.indices(rng, args.batch)
        # Nominal inputs, departures of the deployed model and departures of its current update.
        # Both counterfactual groups retain the recorded visual pair and the same expert target.
        alpha = torch.as_tensor(
            rng.uniform(0, 1, (args.batch, 1)), dtype=torch.float32, device="cuda"
        )
        choice = torch.arange(args.batch, device="cuda") % 3
        dev = torch.where((choice == 1)[:, None], local_parent_errors[ix], current_errors[ix])
        q = train.q[ix] + alpha * dev * (choice != 0)[:, None]
        prediction = train.predict(model, text, ix, q)
        target, valid = train.target[ix], train.valid[ix]
        joint = ((prediction - target) / scales).square().mean(-1)[valid].mean()
        pos, rot = gpu_hand.squared_errors(prediction, target)
        pose = (pos / pose_scale[0] + rot / pose_scale[1])[valid].mean() / 2
        loss = (joint + pose) / 2
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1, error_if_nonfinite=True)
        lr = (
            3e-5
            * min(1, step / 50)
            * (0.15 + 0.85 * (1 + math.cos(math.pi * step / args.updates)) / 2)
        )
        optimizer.param_groups[0]["lr"] = lr
        optimizer.step()
        running += torch.stack([loss.detach(), joint.detach(), pose.detach()])
        updates += 1
        values = {}
        if step % 25 == 0 or step == 1:
            total, motor, spatial = (running / updates).tolist()
            values = dict(
                zip(
                    ("train/loss", "train/action_mse", "train/pose"),
                    (total, motor, spatial),
                    strict=True,
                )
            )
            values["train/lr"] = lr
            updates = 0
            running.zero_()
        if step % args.eval_every == 0 or step == args.updates:
            summary, details = score_replay(val, val.rollout(model, text, args.send), cpu_hand)
            values.update({"validation/" + k: v for k, v in summary.items()})
            (args.run / f"replay-{step:05d}.json").write_text(
                json.dumps(dict(summary=summary, episodes=details), indent=2)
            )
            ranking = (-summary["events"], summary["four_events_xyz_mm"] or math.inf)
            if ranking < best_score:
                best_score = ranking
                atomic_save(
                    dict(config=cfg, model=model.state_dict(), step=parent_step + step),
                    args.run / "policy-best.pt",
                )
                (args.run / "best.json").write_text(
                    json.dumps(
                        dict(update=step, parent_step=parent_step, summary=summary), indent=2
                    )
                )
            atomic_save(
                dict(
                    config=cfg,
                    model=model.state_dict(),
                    step=step,
                    optimizer=optimizer.state_dict(),
                    best_score=best_score,
                ),
                args.run / "training-latest.pt",
            )
            if step != args.updates:
                current_errors = train.rollout(model, text, args.send)
        if values:
            report(step, values)
    atomic_save(
        dict(config=cfg, model=model.state_dict(), step=parent_step + args.updates),
        args.run / "policy-final.pt",
    )
    swanlab.finish()


if __name__ == "__main__":
    main()
