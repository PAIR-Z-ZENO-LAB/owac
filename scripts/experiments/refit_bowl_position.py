"""Refit visual motion differences on paired training approaches while preserving native actions."""

import argparse
import copy
import itertools
import json
import os
from pathlib import Path

import numpy as np
import swanlab
import torch
from scipy.spatial.distance import cdist

from owac.raw.all_bench_distillation import load_inventory
from owac.training.compact_train import atomic_save
from owac.training.fruit_grasp import NeroHand
from owac.training.teleop_recovery import fixed_language, load_export
from owac.training.teleop_recovery_replay import RecoveryBank, event_points, score_replay


def fit_pairs(bank):
    q = bank.q.cpu().numpy()
    scale = bank.sd.cpu().numpy()
    groups = []
    for name, a, n, t, w in zip(
        bank.names, bank.starts, bank.sizes, bank.clocks, bank.thresholds, strict=True
    ):
        ev = event_points(t, q[a : a + n], w)
        groups.append(
            (
                name.split("/")[0],
                [a + np.flatnonzero((t <= end) & (t >= end - 1.5)) for end, _ in ev],
            )
        )
    pairs = []
    for (s, aa), (t, bb) in itertools.combinations(groups, 2):
        if s != t:
            continue
        for a, b in zip(aa, bb, strict=False):
            dist = cdist(q[a] / scale, q[b] / scale)
            for i, j in enumerate(dist.argmin(1)):
                if dist[:, j].argmin() == i:
                    pairs.append((a[i], b[j], dist[i, j]))
    rows = np.asarray(pairs)
    weights = np.exp(-0.5 * (rows[:, 2] / np.median(rows[:, 2])) ** 2)
    return rows[:, :2].astype(int), weights / weights.sum()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for k in ("experiment", "parent", "urdf", "run"):
        parser.add_argument(k, type=Path)
    args = parser.parse_args()
    args.run.mkdir(exist_ok=True, parents=True)
    root = args.experiment
    torch.manual_seed(20261124)
    torch.set_num_threads(2)
    model, step = load_export(args.parent)
    cfg = model.cfg
    model.requires_grad_(False)
    model.motion_correction.requires_grad_(True)
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    fixed = copy.deepcopy(ema)
    txt = fixed_language(model)
    src, _ = load_inventory(root / "aligned/inventory.json")
    tr = json.loads((root / "field-tracking.json").read_text())
    train, val = [
        RecoveryBank([s for s in src if s.split == split], root / "visual", cfg, tr)
        for split in ("train", "validation")
    ]
    for b in (train, val):
        b.configure_output(cfg)
        b.load_spatial(root / "spatial")
        b.history_rows(cfg["model"]["state_history_lags"])
    pairs, prob = fit_pairs(train)
    gpu = NeroHand(args.urdf, "cuda")
    cpu = NeroHand(args.urdf, "cpu")
    original = json.loads((root.parent / "runs/owac/config.json").read_text())["statistics"]
    scale = (train.tensor(original["action_q99"]) - train.tensor(original["action_q01"]) + 1e-6) / 2
    with torch.no_grad():
        err = train.rollout(fixed, txt, 4)
        pos = gpu.pose(train.target)[0]
        now = gpu.pose(train.q)[0]
        move = pos - now[:, None]
        aa, bb = pairs.T
        magnitude = (move[bb] - move[aa]).square().sum(-1).mean()
        nominal_move = move.square().sum(-1).mean()
    orig_pairs = json.loads(
        (root.parent / "grasp-comparison-20261124/visual-response.json").read_text()
    )["pairs"]
    va, vb = [], []
    expected = []
    for row in orig_pairs:
        if row["protocol"] == "approach":
            aa, bb = [
                val.starts[val.names.index(ep)] + i
                for ep, i in zip(row["episodes"], row["frames"], strict=True)
            ]
            va.append(aa)
            vb.append(bb)
            expected.append(row["reference_xy_mm"])
    va, vb = (torch.tensor(x, device="cuda") for x in (va, vb))
    expected = val.tensor(expected)

    def swapped(b, m, a, c):
        ix = torch.cat([a, c, a, c])
        fixed_ix = torch.cat([a, a, c, c])
        native = b.predict(m, txt, ix, b.q[fixed_ix], b.q[b.history_index[fixed_ix]])
        pp = gpu.pose(native.clamp(b.low, b.high))[0]
        pa, pb, pc, pd = pp.chunk(4)
        return (pb - pa + pd - pc) / 2

    exec_cfg = json.loads((root / "execution-comparison.json").read_text())
    gaps = exec_cfg["gaps"]

    def evaluate(m):
        with torch.no_grad():
            shift = swapped(val, m, va, vb)[:, -1, :2] * 1000
            projection = (shift * expected).sum(1) / expected.norm(dim=1)
            direction = int((projection > 0).sum())
            runs = []
            for gap in gaps:
                summary, episodes = score_replay(val, val.rollout(m, txt, 4, gaps=gap), cpu)
                runs.append(dict(summary=summary, episodes=episodes))
        keys = (
            "first_grasp_xy_mm",
            "four_events_xyz_mm",
            "four_events_xy_mm",
            "full_hand_mm",
            "full_gripper_mm",
            "four_events_timing_mae_s",
        )
        values = {k: float(np.mean([r["summary"][k] for r in runs])) for k in keys}
        values.update(
            events=sum(r["summary"]["events"] for r in runs),
            visual_agrees=direction,
            visual_median_shift_mm=float(np.median(shift.norm(dim=1).cpu().numpy())),
        )
        return values, runs, projection.cpu().tolist()

    baseline, _, _ = evaluate(fixed)
    spec = dict(
        source_step=step,
        training_pairs=len(pairs),
        train_episodes=train.names,
        validation_episodes=val.names,
        updates=800,
        trainable="temporal action correction",
        task=(
            "match local expert motion differences between paired views "
            "and retain native trajectories"
        ),
        selection=(
            "retain event coverage and improve first-grasp, four-event position "
            "and visual direction on development recordings"
        ),
    )
    cfg["position_training"] = spec
    (args.run / "config.json").write_text(json.dumps(cfg, indent=2))
    sw = swanlab.init(
        workspace=os.environ.get("SWANLAB_WORKSPACE"),
        mode=os.environ.get("SWANLAB_MODE", "local"),
        project=os.environ.get("SWANLAB_PROJECT", "OWAC"),
        id="bowl-local-position-20261124",
        name="OWAC | bowl | paired position correction | local RTX 5090",
        group="real-bowl-20261123",
        job_type="action-policy",
        public=False,
        resume="allow",
        config=spec,
        logdir=str(args.run / "swanlog"),
    )
    (args.run / "swanlab.json").write_text(json.dumps(dict(url=sw.url)))
    opt = torch.optim.AdamW(model.motion_correction.parameters(), lr=4e-5)
    best = (1, float("inf"))
    best_values = baseline
    for i in range(1, 801):
        rng = np.random.default_rng(20261124 + i)
        choose = pairs[rng.choice(len(pairs), 24, p=prob)]
        a, c = [torch.tensor(k, device="cuda") for k in choose.T]
        visual_delta = swapped(train, model, a, c)
        pair_loss = (visual_delta - (move[c] - move[a])).square().sum(-1).mean() / magnitude
        ix = train.indices(rng, 48)
        h = train.history_index[ix]
        frac = train.tensor(rng.random((len(ix), 1)))
        q = train.q[ix] + frac * err[ix]
        past = train.q[h] + frac[:, None] * err[h]
        pred = train.predict(model, txt, ix, q, past)
        with torch.no_grad():
            anchor = train.predict(fixed, txt, ix, q, past)
        ok = train.valid[ix]
        target = train.target[ix]
        preserve = ((pred - anchor) / scale).square().mean(-1)[ok].mean()
        imitate = ((pred - target) / scale).square().mean(-1)[ok].mean()
        geom = (gpu.pose(pred)[0] - pos[ix]).square().sum(-1)[ok].mean() / nominal_move
        loss = (pair_loss + preserve + imitate + geom) / 4
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.motion_correction.parameters(), 1, error_if_nonfinite=True
        )
        opt.step()
        with torch.no_grad():
            torch._foreach_lerp_(
                list(ema.motion_correction.parameters()),
                list(model.motion_correction.parameters()),
                0.025,
            )
        values = {}
        if i % 25 == 0 or i == 1:
            values = {
                "train/loss": float(loss.detach()),
                "train/paired_motion": float(pair_loss.detach()),
                "train/native_anchor": float(preserve.detach()),
            }
        if i % 100 == 0:
            metrics, runs, projections = evaluate(ema)
            values.update({"validation/" + k: v for k, v in metrics.items()})
            (args.run / f"replay-{i:05d}.json").write_text(
                json.dumps(
                    dict(summary=metrics, protocols=runs, projections_mm=projections), indent=2
                )
            )
            improves = (
                metrics["events"] >= baseline["events"]
                and metrics["visual_agrees"] > baseline["visual_agrees"]
                and metrics["first_grasp_xy_mm"] < baseline["first_grasp_xy_mm"]
                and metrics["four_events_xyz_mm"] < baseline["four_events_xyz_mm"]
            )
            ranking = (0 if improves else 1, metrics["four_events_xyz_mm"])
            if ranking < best:
                best = ranking
                best_values = metrics
                export_cfg = copy.deepcopy(cfg)
                atomic_save(
                    dict(config=export_cfg, model=ema.state_dict(), step=step + i),
                    args.run / "policy-best.pt",
                )
                (args.run / "best.json").write_text(
                    json.dumps(
                        dict(
                            parent_step=step,
                            update=i,
                            dominates_parent=improves,
                            summary=best_values,
                            baseline=baseline,
                        ),
                        indent=2,
                    )
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
        if values:
            swanlab.log(values, step=i)
            with (args.run / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(dict(step=i, metrics=values)) + "\n")
            print(i, json.dumps(values), flush=True)
    atomic_save(
        dict(config=cfg, model=ema.state_dict(), step=step + 800), args.run / "policy-final.pt"
    )
    swanlab.finish()


if __name__ == "__main__":
    main()
