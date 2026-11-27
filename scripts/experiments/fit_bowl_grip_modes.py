"""Fit jaw persistence, measurement likelihoods and conditional widths from training demos."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from owac.raw.all_bench_distillation import load_inventory
from owac.training.compact_train import atomic_save
from owac.training.fruit_grasp import NeroHand
from owac.training.teleop_recovery import fixed_language, load_export
from owac.training.teleop_recovery_replay import RecoveryBank, score_replay


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("experiment", "parent", "urdf", "run"):
        p.add_argument(k, type=Path)
    args = p.parse_args()
    args.run.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(2)
    model, step = load_export(args.parent)
    cfg = model.cfg
    txt = fixed_language(model)
    hand = NeroHand(args.urdf, "cpu")
    root = args.experiment
    src, _ = load_inventory(root / "aligned/inventory.json")
    track = json.loads((root / "field-tracking.json").read_text())

    def load(split):
        b = RecoveryBank([s for s in src if s.split == split], root / "visual", cfg, track)
        b.configure_output(cfg)
        b.load_spatial(root / "spatial")
        b.history_rows(cfg["model"]["state_history_lags"])
        return b

    b = load("train")
    predicted = []
    with torch.no_grad():
        err = b.rollout(model, txt, 4)
        for e in (torch.zeros_like(err), err):
            for ix in torch.arange(len(b.q), device="cuda").split(96):
                hist = b.history_index[ix]
                pred = b.predict(model, txt, ix, b.q[ix] + e[ix], b.q[hist] + e[hist])[:, :, 7]
                predicted.append(pred.cpu().numpy())
    native = b.target[:, :, 7].cpu().numpy()
    classes = native > np.repeat(b.thresholds, b.sizes)[:, None]
    widths = b.q[:, 7].cpu().numpy()

    def moments(v, c):
        return [[float(fn(v[c == k])) for k in (0, 1)] for fn in (np.mean, np.var)]

    q_mean, q_var = moments(widths, classes[:, 0])
    counts = np.zeros((2, 2))
    for a, n in zip(b.starts, b.sizes, strict=True):
        c = classes[a : a + n, 0].astype(int)
        np.add.at(counts, (c[:-1], c[1:]), 1)
    p01, p10 = counts[0, 1] / counts[0].sum(), counts[1, 0] / counts[1].sum()
    keep = b.valid.cpu().numpy()
    y = np.tile(native, (2, 1))[np.tile(keep, (2, 1))]
    c = np.tile(classes, (2, 1))[np.tile(keep, (2, 1))]
    x = np.concatenate(predicted)[np.tile(keep, (2, 1))]
    x_mean, x_var = moments(x, c)
    slope, intercept = np.array([np.polyfit(x[c == i], y[c == i], 1) for i in (0, 1)]).T.tolist()
    low, high = [[float(fn(y[c == i])) for i in (0, 1)] for fn in (np.min, np.max)]
    fit = [[p01, p10], q_mean, q_var, x_mean, x_var, slope, intercept, low, high]
    model.set_jaw_fit(fit)
    del b
    b = load("validation")
    gaps = json.loads((root / "execution-comparison.json").read_text())["gaps"]
    runs = {}
    for n in (4, 5, 6, 12):
        protocols = []
        for gap in gaps:
            err = b.rollout(model, txt, n, gaps=gap)
            s, eps = score_replay(b, err, hand)
            protocols.append(dict(summary=s, episodes=eps))
        keys = (
            "first_grasp_xy_mm",
            "four_events_xyz_mm",
            "four_events_xy_mm",
            "full_hand_mm",
            "full_gripper_mm",
            "four_events_timing_mae_s",
        )
        scores = {k: float(np.mean([q["summary"][k] for q in protocols])) for k in keys}
        runs[str(n)] = dict(
            events_across_schedules=sum(q["summary"]["events"] for q in protocols),
            means=scores,
            protocols=protocols,
        )
        print(n, scores, flush=True)
    proof = dict(
        fit=fit,
        training_episodes=[s.group_id for s in src if s.split == "train"],
        observations=counts.sum().item(),
        runs=runs,
    )
    (args.run / "validation.json").write_text(json.dumps(proof, indent=2))
    atomic_save(dict(config=cfg, model=model.state_dict(), step=step), args.run / "policy.pt")


if __name__ == "__main__":
    main()
