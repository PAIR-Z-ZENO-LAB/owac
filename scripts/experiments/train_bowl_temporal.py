"""Fit a causal motion correction on spatial OWAC using measured and replayed trajectories."""

import argparse
import copy
import json
import math
import os
from pathlib import Path

import numpy as np
import swanlab
import torch
from torch.nn import functional as F

from owac.frameworks.raw_compact.factory import make_policy
from owac.raw.all_bench_distillation import load_inventory
from owac.training.compact_train import atomic_save
from owac.training.fruit_grasp import NeroHand
from owac.training.teleop_recovery import fixed_language, predict_cached
from owac.training.teleop_recovery_replay import RecoveryBank, score_replay


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("experiment", "parent", "urdf", "run"):
        p.add_argument(arg, type=Path)
    args = p.parse_args()
    args.run.mkdir(exist_ok=True, parents=True)
    torch.manual_seed(20261124)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    parent = torch.load(args.parent, weights_only=False, map_location="cpu", mmap=True)
    cfg = parent["config"]
    start = parent["step"]
    cfg["model"]["kind"] = "corp_temporal_spatial_regression"
    lags = [1, 3, 6, 12, 24]
    cfg["model"]["state_history_lags"] = lags
    model = make_policy(cfg)
    info = model.load_state_dict(parent["model"], strict=False)
    assert not info.unexpected_keys and all(
        k.startswith("motion_correction.") for k in info.missing_keys
    )
    del parent
    model = model.cuda().eval().requires_grad_(False)
    model.motion_correction.requires_grad_(True)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    text = fixed_language(model)
    src, _ = load_inventory(args.experiment / "aligned/inventory.json")
    tracking = json.loads((args.experiment / "field-tracking.json").read_text())
    train, val = [
        RecoveryBank([s for s in src if s.split == spl], args.experiment / "visual", cfg, tracking)
        for spl in ("train", "validation")
    ]
    for bank in (train, val):
        bank.configure_output(cfg)
        bank.load_spatial(args.experiment / "spatial")
        bank.history_rows(lags)
    cpu_hand, gpu_hand = NeroHand(args.urdf, "cpu"), NeroHand(args.urdf, "cuda")
    original = json.loads((args.experiment.parent / "runs/owac/config.json").read_text())[
        "statistics"
    ]
    scales = (
        train.tensor(original["action_q99"]) - train.tensor(original["action_q01"]) + 1e-6
    ) / 2
    half = train.tensor(np.repeat(train.thresholds, train.sizes))
    w = train.target[:, :, 7][train.valid]
    opened = (train.target[:, :, 7] > half[:, None])[train.valid]
    width_std = (
        (w - torch.where(opened, w[opened].mean(), w[~opened].mean())).square().mean().sqrt()
    )
    with torch.no_grad():
        parent_errors = train.rollout(ema, text, 5)
        current = parent_errors
        pos, rot = gpu_hand.squared_errors(train.q[:, None].expand_as(train.target), train.target)
        ps, rs = pos[train.valid].mean(), rot[train.valid].mean()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            visual = torch.cat([model.visual_plan(m) for m in train.maps.split(128)])
    opt = torch.optim.AdamW(model.motion_correction.parameters(), lr=0.00015, weight_decay=0.01)
    spec = dict(
        parent_step=start,
        updates=4000,
        send=5,
        lag_frames=lags,
        train_episodes=train.names,
        development_episodes=val.names,
        objective="motor, hand pose and gripper mode at recorded expert commands",
        perturbations="Time-consistent errors from parent and current policy replay",
        width_scale_m=float(width_std),
        selection="event coverage then 3D error over both picks and releases",
    )
    cfg["temporal_training"] = spec
    (args.run / "config.json").write_text(json.dumps(cfg, indent=2))
    run = swanlab.init(
        workspace=os.environ.get("SWANLAB_WORKSPACE"),
        mode=os.environ.get("SWANLAB_MODE", "local"),
        project=os.environ.get("SWANLAB_PROJECT", "OWAC"),
        id="bowl-local-motion-history-20261124",
        name="OWAC | bowl | measured motion feedback | local RTX 5090",
        public=False,
        resume="allow",
        group="real-bowl-20261123",
        job_type="action-policy",
        config=spec,
        logdir=str(args.run / "swanlog"),
    )
    (args.run / "swanlab.json").write_text(json.dumps(dict(url=run.url)))
    best = (0, math.inf)
    running = 0.0
    count = 0
    for i in range(1, spec["updates"] + 1):
        rng = np.random.default_rng(20261124 + i)
        ix = train.indices(rng, 96)
        hix = train.history_index[ix]
        mode = torch.arange(len(ix), device="cuda") % 3
        alpha = train.tensor(rng.random((len(ix), 1))) * (mode != 0)[:, None]
        q = train.q[ix] + alpha * torch.where((mode == 1)[:, None], parent_errors[ix], current[ix])
        past = train.q[hix] + alpha[:, None] * torch.where(
            (mode == 1)[:, None, None], parent_errors[hix], current[hix]
        )
        q = (q - train.mean) / train.sd
        past = (past - train.mean) / train.sd
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            base = predict_cached(model, train.visual[ix], q, text)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            norm = model.correct(base, visual[ix], train.maps[ix], q, past)
        cmd = (norm + 1) * train.half + train.zero
        target, ok = train.target[ix], train.valid[ix]
        mse = ((cmd - target) / scales).square().mean(-1)[ok].mean()
        pos, rot = gpu_hand.squared_errors(cmd, target)
        pose = (pos / ps + rot / rs)[ok].mean() / 2
        margin = (cmd[:, :, 7] - half[ix, None]) / width_std
        mode_loss = F.binary_cross_entropy_with_logits(
            margin[ok], (target[:, :, 7] > half[ix, None])[ok].float()
        )
        loss = (mse + pose + mode_loss) / 3
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.motion_correction.parameters(), 1, error_if_nonfinite=True
        )
        lr = (
            0.00015
            * min(1, i / 60)
            * (0.08 + 0.92 * (1 + math.cos(math.pi * i / spec["updates"])) / 2)
        )
        opt.param_groups[0]["lr"] = lr
        opt.step()
        with torch.no_grad():
            torch._foreach_lerp_(
                list(ema.motion_correction.parameters()),
                list(model.motion_correction.parameters()),
                0.01,
            )
        running += float(loss.detach())
        count += 1
        values = {}
        if i % 25 == 0 or i == 1:
            values = {"train/loss": running / count, "train/lr": lr}
            running = 0.0
            count = 0
        if i % 400 == 0:
            summary, details = score_replay(val, val.rollout(ema, text, 5), cpu_hand)
            values.update({"validation/" + k: v for k, v in summary.items() if v is not None})
            (args.run / f"replay-{i:05d}.json").write_text(
                json.dumps(dict(summary=summary, episodes=details), indent=2)
            )
            score = (-summary["events"], summary["four_events_xyz_mm"] or math.inf)
            if score < best:
                best = score
                atomic_save(
                    dict(config=cfg, model=ema.state_dict(), step=start + i),
                    args.run / "policy-best.pt",
                )
                (args.run / "best.json").write_text(
                    json.dumps(dict(parent_step=start, update=i, summary=summary), indent=2)
                )
            atomic_save(
                dict(
                    config=cfg,
                    model=model.state_dict(),
                    ema=ema.state_dict(),
                    optimizer=opt.state_dict(),
                    step=i,
                    best=best,
                ),
                args.run / "training-latest.pt",
            )
            if i % 800 == 0 and i < spec["updates"]:
                current = train.rollout(ema, text, 5)
        if values:
            swanlab.log(values, step=i)
            with (args.run / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(dict(step=i, metrics=values), allow_nan=False) + "\n")
            print(i, json.dumps(values), flush=True)
    atomic_save(
        dict(config=cfg, model=ema.state_dict(), step=start + spec["updates"]),
        args.run / "policy-final.pt",
    )
    swanlab.finish()


if __name__ == "__main__":
    main()
