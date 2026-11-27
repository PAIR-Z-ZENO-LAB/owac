"""Train a spatial RAW action readout with the local absolute-target OWAC policy."""

import argparse
import copy
import json
import math
import os
from pathlib import Path

import numpy as np
import swanlab
import torch

from owac.frameworks.raw_compact.factory import make_policy
from owac.raw.all_bench_distillation import load_inventory
from owac.training.compact_train import atomic_save
from owac.training.fruit_grasp import NeroHand
from owac.training.teleop_recovery import fixed_language, predict_cached
from owac.training.teleop_recovery_replay import RecoveryBank, score_replay


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("parent", "experiment", "urdf", "run"):
        p.add_argument(arg, type=Path)
    p.add_argument("--updates", type=int, default=4800)
    p.add_argument("--send", type=int, default=5)
    p.add_argument("--id", default="bowl-local-spatial-feedback-20261124")
    args = p.parse_args()
    args.run.mkdir(exist_ok=True, parents=True)
    torch.manual_seed(20261124)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    payload = torch.load(args.parent, weights_only=False, mmap=True, map_location="cpu")
    cfg = payload["config"]
    assert not any(cfg["observation"]["relative_action_mask"])
    cfg["model"]["kind"] = "corp_spatial_regression"
    model = make_policy(cfg)
    info = model.load_state_dict(payload["model"], strict=False)
    assert not info.unexpected_keys and all(
        k.startswith(("spatial_", "compress.", "base_logit")) for k in info.missing_keys
    )
    model.compress.load_state_dict(
        torch.load(args.experiment / "spatial/projection.pt", weights_only=True)
    )
    model = model.cuda().eval().requires_grad_(False)
    parent_step = payload["step"]
    del payload
    text = fixed_language(model)
    trainable = [model.base_logit]
    model.base_logit.requires_grad_(True)
    for block in (model.spatial_features, model.spatial_actions):
        block.requires_grad_(True)
        trainable.extend(block.parameters())
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    ema_params = [
        ema.base_logit,
        *ema.spatial_features.parameters(),
        *ema.spatial_actions.parameters(),
    ]
    src, _ = load_inventory(args.experiment / "aligned/inventory.json")
    tracking = json.loads((args.experiment / "field-tracking.json").read_text())
    train, val = [
        RecoveryBank(
            [s for s in src if s.split == split], args.experiment / "visual", cfg, tracking
        )
        for split in ("train", "validation")
    ]
    for bank in (train, val):
        bank.configure_output(cfg)
        bank.load_spatial(args.experiment / "spatial")
    gpu_hand, cpu_hand = NeroHand(args.urdf, "cuda"), NeroHand(args.urdf, "cpu")
    with torch.no_grad():
        # Normalize Cartesian and angular losses by the spread of training targets.
        pos, rot = gpu_hand.pose(train.target[:, 0])
        posvar = pos.var(0).sum()
        rotvar = rot.var(0).sum() / 2
        moving = (train.target[:, 1:] - train.target[:, :-1])[train.valid[:, 1:]]
        velocity_scale = moving.square().mean(0).sqrt()
    optimizer = torch.optim.AdamW(trainable, lr=0.0003, weight_decay=0.01)
    original = json.loads((args.experiment / "absolute/best.json").read_text())
    spec = dict(
        source_step=parent_step,
        fit_episodes=train.names,
        development_episodes=val.names,
        updates=args.updates,
        batch=96,
        send=args.send,
        architecture="OWAC resampler and a dense two-view readout of its RAW encoder",
        supervision="Bounded expert targets over the whole episode and their forward kinematics",
        selection="Event count then mean 3D distance at two picks and releases",
        absolute_parent=original["summary"],
        physical_trials=0,
    )
    cfg["spatial_training"] = spec
    (args.run / "config.json").write_text(json.dumps(cfg, indent=2))
    run = swanlab.init(
        workspace=os.environ.get("SWANLAB_WORKSPACE"),
        mode=os.environ.get("SWANLAB_MODE", "local"),
        project=os.environ.get("SWANLAB_PROJECT", "OWAC"),
        id=args.id,
        public=False,
        resume="allow",
        name="OWAC | bowl | spatial action readout | local RTX 5090",
        group="real-bowl-20261123",
        job_type="action-policy",
        config=spec,
        logdir=str(args.run / "swanlog"),
    )
    (args.run / "swanlab.json").write_text(json.dumps(dict(url=run.url)))

    def report(step, values):
        values = {k: v for k, v in values.items() if v is not None}
        swanlab.log(values, step=step)
        with (args.run / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(dict(step=step, metrics=values), allow_nan=False) + "\n")
        print(step, json.dumps(values), flush=True)

    best = (0, math.inf)
    # States sampled from the actual error of the frozen parent in this replay.
    parent_logit = model.base_logit.detach().clone()
    with torch.no_grad():
        model.base_logit.fill_(100)
        errors = train.rollout(model, text, args.send)
        model.base_logit.copy_(parent_logit)
    rolling = 0.0
    count = 0
    for step in range(1, args.updates + 1):
        rng = np.random.default_rng(20261124 + step)
        idx = train.indices(rng, 96)
        target, ok = train.target[idx], train.valid[idx]
        q = train.q[idx] + errors[idx] * train.tensor(rng.random((len(idx), 1)))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            base = predict_cached(model, train.visual[idx], (q - train.mean) / train.sd, text)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            visual = model.visual_plan(train.maps[idx])
        blended = model.combine(base, visual)
        reference = (target - train.zero) / train.half - 1
        # The independent visual estimate must also predict the motor goals correctly.
        mse = ((visual.float() - reference).square() + (blended - reference).square())[
            ok
        ].mean() / 2
        command = (blended + 1) * train.half + train.zero
        p_err, r_err = gpu_hand.squared_errors(command, target)
        pose = (p_err / posvar + r_err / rotvar)[ok].mean() / 2
        diff = torch.diff(command - target, dim=1) / velocity_scale
        derivative = diff.square()[ok[:, 1:]].mean()
        loss = mse + pose + derivative / target.shape[1]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1, error_if_nonfinite=True)
        lr = (
            0.0003
            * min(1, step / 80)
            * (0.08 + 0.92 * (1 + math.cos(math.pi * step / args.updates)) / 2)
        )
        optimizer.param_groups[0]["lr"] = lr
        optimizer.step()
        with torch.no_grad():
            torch._foreach_lerp_(ema_params, trainable, 0.01)
        rolling += float(loss.detach())
        count += 1
        values = {}
        if step == 1 or step % 25 == 0:
            values = {
                "train/loss": rolling / count,
                "train/lr": lr,
                "train/base_share": float(model.base_logit.detach().sigmoid().mean()),
            }
            rolling = 0.0
            count = 0
        if step % 400 == 0 or step == args.updates:
            summary, episodes = score_replay(val, val.rollout(ema, text, args.send), cpu_hand)
            values.update({"validation/" + k: v for k, v in summary.items()})
            (args.run / f"replay-{step:05d}.json").write_text(
                json.dumps(dict(summary=summary, episodes=episodes), indent=2)
            )
            score = (-summary["events"], summary["four_events_xyz_mm"] or math.inf)
            if score < best:
                best = score
                atomic_save(
                    dict(config=cfg, step=parent_step + step, model=ema.state_dict()),
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
                    step=step,
                    model=model.state_dict(),
                    ema=ema.state_dict(),
                    optimizer=optimizer.state_dict(),
                    best=best,
                ),
                args.run / "training-latest.pt",
            )
        if values:
            report(step, values)
    atomic_save(
        dict(config=cfg, step=parent_step + args.updates, model=ema.state_dict()),
        args.run / "policy-final.pt",
    )
    swanlab.finish()


if __name__ == "__main__":
    main()
