"""Train a conditional jaw decoder using recorded open/closed modes and actual widths."""

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
from owac.training.teleop_recovery import fixed_language
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
    payload = torch.load(args.parent, weights_only=False, map_location="cpu", mmap=True)
    cfg = payload["config"]
    cfg["model"]["conditional_aperture"] = True
    model = make_policy(cfg)
    info = model.load_state_dict(payload["model"], strict=False)
    assert not info.unexpected_keys and all(k.startswith("aperture.") for k in info.missing_keys)
    start = payload["step"]
    del payload
    model = model.cuda().eval().requires_grad_(False)
    model.aperture.requires_grad_(True)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    text = fixed_language(model)
    src, _ = load_inventory(args.experiment / "aligned/inventory.json")
    track = json.loads((args.experiment / "field-tracking.json").read_text())
    train, val = [
        RecoveryBank([s for s in src if s.split == splt], args.experiment / "visual", cfg, track)
        for splt in ("train", "validation")
    ]
    for bank in (train, val):
        bank.configure_output(cfg)
        bank.load_spatial(args.experiment / "spatial")
    parent_aperture = model.aperture
    model.aperture = None
    with torch.inference_mode():
        departure = train.rollout(model, text, 5)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            visual = torch.cat([model.visual_plan(m) for m in train.maps.split(128)])
    model.aperture = parent_aperture
    visual = visual.clone()
    half = train.tensor(np.repeat(train.thresholds, train.sizes))
    hand = NeroHand(args.urdf, "cpu")
    opt = torch.optim.AdamW(model.aperture.parameters(), lr=0.0005, weight_decay=0.01)
    spec = dict(
        parent_step=start,
        updates=3200,
        send=5,
        train_episodes=train.names,
        development_episodes=val.names,
        head="Open/closed classifier and two width regressions using arm state and both RAW views",
        objective="whole episode motor widths, oversampling human grip transitions",
        selection="event count then mean 3D distance at two picks and releases",
    )
    cfg["aperture_training"] = spec
    (args.run / "config.json").write_text(json.dumps(cfg, indent=2))
    run = swanlab.init(
        workspace=os.environ.get("SWANLAB_WORKSPACE"),
        mode=os.environ.get("SWANLAB_MODE", "local"),
        project=os.environ.get("SWANLAB_PROJECT", "OWAC"),
        id="bowl-local-aperture-20261124",
        name="OWAC | bowl | conditional grasp and release | local RTX 5090",
        group="real-bowl-20261123",
        public=False,
        resume="allow",
        job_type="action-policy",
        config=spec,
        logdir=str(args.run / "swanlog"),
    )
    (args.run / "swanlab.json").write_text(json.dumps(dict(url=run.url)))
    best = (0, math.inf)
    tot = 0.0
    count = 0
    for i in range(1, spec["updates"] + 1):
        rng = np.random.default_rng(20261124 + i)
        ix = train.indices(rng, 128)
        # Alternate recorded arm poses and departures seen in parent-policy replay.
        dev = (
            train.tensor(rng.random((len(ix), 1)))
            * (torch.arange(len(ix), device="cuda") % 2)[:, None]
        )
        q = (train.q[ix] + dev * departure[ix] - train.mean) / train.sd
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, lo, hi = model.aperture_prediction(train.maps[ix], q, visual[ix]).unbind(-1)
        width = train.target[ix, :, 7]
        open_mode = width > half[ix, None]
        norm = (width - train.zero[7]) / train.half[7] - 1
        ok = train.valid[ix]
        loss = F.binary_cross_entropy_with_logits(logits[ok], open_mode[ok].float())
        loss += F.mse_loss(torch.where(open_mode, hi, lo)[ok], norm[ok])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.aperture.parameters(), 1, error_if_nonfinite=True)
        lr = (
            0.0005
            * min(1, i / 50)
            * (0.08 + 0.92 * (1 + math.cos(math.pi * i / spec["updates"])) / 2)
        )
        opt.param_groups[0]["lr"] = lr
        opt.step()
        with torch.no_grad():
            torch._foreach_lerp_(
                list(ema.aperture.parameters()), list(model.aperture.parameters()), 0.015
            )
        tot += float(loss.detach())
        count += 1
        values = {}
        if i % 25 == 0 or i == 1:
            values = dict(zip(("train/loss", "train/lr"), (tot / count, lr), strict=True))
            tot = 0.0
            count = 0
        if i % 400 == 0:
            summary, details = score_replay(val, val.rollout(ema, text, 5), hand)
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
                    model=model.state_dict(),
                    ema=ema.state_dict(),
                    config=cfg,
                    optimizer=opt.state_dict(),
                    step=i,
                    best=best,
                ),
                args.run / "training-latest.pt",
            )
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
