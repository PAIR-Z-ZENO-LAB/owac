"""Select the joint smoothing penalty on demonstration command tracking and its derivatives."""

import argparse
import json
from pathlib import Path

import numpy as np

from owac.runtime.bowl_postprocess import JointPlanSmoother


def derivatives(x, t):
    dt = np.diff(t)
    velocity = np.diff(x, axis=0) / dt[:, None]
    accel = np.diff(velocity, axis=0) / ((dt[1:] + dt[:-1]) / 2)[:, None]
    return x, velocity, accel


def replay(plans, rows, t, q, penalty, limits):
    smoother = JointPlanSmoother(penalty, limits)
    return np.stack(
        [
            smoother.step(plan, measured, now, row)
            for plan, row, now, measured in zip(plans, rows, t, q, strict=True)
        ]
    )


def measures(joints, times, rows):
    _, velocity, accel = derivatives(np.rad2deg(joints), times)
    jump = np.abs(np.diff(np.rad2deg(joints), axis=0)).max(1)
    return dict(
        joint_step_p95_deg=float(np.percentile(jump, 95)),
        boundary_step_p95_deg=float(np.percentile(jump[rows[1:] == 0], 95)),
        max_step_deg=float(np.max(jump)),
        axis_speed_p95_deg_s=float(np.percentile(np.abs(velocity).max(1), 95)),
        axis_accel_p95_deg_s2=float(np.percentile(np.abs(accel).max(1), 95)),
    )


def main(root, tracking):
    features = root / "features"
    travel = json.loads(tracking.read_text())
    limits = [travel["low"][:7], travel["high"][:7]]
    episodes = {}
    for source in sorted(features.glob("bowl*.npz")):
        with np.load(source) as ep:
            all_t, q, target = ep["t"], ep["q"], ep["command"]
        native = np.load(features / (source.stem + "-plans.npy"), mmap_mode="r")
        starts = np.arange(0, len(q) - 4, 5)
        index = (starts[:, None] + np.arange(4)).ravel()
        row = np.tile(np.arange(4), len(starts))
        plans = native[np.repeat(starts, 4), :4, :7]
        episodes[source.stem] = (
            plans,
            row,
            all_t[index],
            q[index],
            np.clip(target[index, :7], *limits),
        )
    train = [k for k in episodes if not k.endswith(("005", "010"))]
    val = [k for k in episodes if k not in train]
    scales = [
        np.concatenate([derivatives(episodes[k][-1], episodes[k][2])[i] for k in train]).std(0)
        for i in range(3)
    ]
    runs = {}
    for penalty in (0.0, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0):
        result = {}
        for name, (plan, row, t, q, label) in episodes.items():
            cmd = replay(plan, row, t, q, penalty, limits)
            losses = [
                float(np.sqrt(np.mean(((a - b) / scale) ** 2)))
                for a, b, scale in zip(
                    derivatives(cmd, t), derivatives(label, t), scales, strict=True
                )
            ]
            delta = np.rad2deg(cmd - label)
            result[name] = dict(
                normalized_rmse=losses,
                joint_mae_deg=float(np.abs(delta).mean()),
                demo=measures(label, t, row),
                **measures(cmd, t, row),
            )
        runs[penalty] = dict(
            train_loss=float(np.mean([result[k]["normalized_rmse"] for k in train])),
            episodes=result,
        )
    chosen = min(runs, key=lambda k: runs[k]["train_loss"])
    print(
        "smoothing", [(k, round(r["train_loss"], 4)) for k, r in runs.items()], chosen, flush=True
    )
    (root / "smoothing-fit.json").write_text(
        json.dumps(
            dict(
                chosen=chosen, train=train, validation=val, candidates=runs, joint_limits_rad=limits
            ),
            indent=2,
        )
    )
    release = json.loads((root / "release-profile.json").read_text())
    release.update(joint_penalty=chosen, joint_limits_rad=limits)
    (root / "bowl-postprocess.json").write_text(json.dumps(release, indent=2))
    if not list((root / "field").glob("commands*.jsonl")):
        return
    rows = [
        json.loads(s)
        for s in next((root / "field").glob("commands*.jsonl")).read_text().splitlines()
    ]
    t = (np.array([r["time_ns"] for r in rows]) - rows[0]["time_ns"]) / 1e9
    q, raw = (np.array([r[k] for r in rows]) for k in ("feedback_rad_m", "model_rad_m"))
    index = np.array([r["row"] for r in rows])
    native = np.array([r["issued_rad_m"] for r in rows])
    chunks = [raw[i - irow : i - irow + 4, :7] for i, irow in enumerate(index)]
    filtered = replay(chunks, index, t, q, chosen, limits)
    field = dict(
        native=measures(native[:, :7], t, index),
        bounded_native=measures(np.clip(native[:, :7], *limits), t, index),
        smoothed=measures(filtered, t, index),
        mean_abs_change_deg=float(np.abs(np.rad2deg(filtered - native[:, :7])).mean()),
        notes="803 recorded commands; last paused partial chunk has only three recorded rows",
    )
    (root / "field-smoothing.json").write_text(json.dumps(field, indent=2))
    np.savez_compressed(
        root / "field-smoothing.npz", t=t, model=raw, issued=native, feedback=q, smoothed=filtered
    )
    print("field", field, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment", type=Path)
    p.add_argument("tracking", type=Path)
    args = p.parse_args()
    main(args.experiment, args.tracking)
