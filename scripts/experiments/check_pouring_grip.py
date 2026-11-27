"""Fit grip separation on training demos and inject identifiable faults into held-out telemetry."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from owac.raw.all_bench_distillation import load_inventory
from owac.runtime.pouring_grip import PouringGrip
from owac.training.fruit_grasp import NeroHand
from owac.training.pouring_feedback_data import opening_strokes


def recorded(src, hand):
    with np.load(src.path) as data:
        q, u, t = (data[k] for k in ("state", "action", "observation_ns"))
    _, rot = hand.pose(torch.from_numpy(q))
    axis = rot.numpy()[:, 2, :]
    strokes = opening_strokes(u[:, 7], np.quantile(u[:, 7], [0.1, 0.95]).mean())
    (opened, grasp), (release, shut) = strokes
    return dict(
        name=src.group_id,
        q=q,
        u=u,
        t=t,
        axis=axis,
        opened=opened,
        grasp=grasp,
        release=release,
        shut=shut,
    )


def fit(episodes, hand):
    ref = np.stack([e["axis"][e["grasp"]] for e in episodes])
    rows = []
    for ep in episodes:
        q, u = ep["q"], ep["u"]
        g, r = ep["grasp"], ep["release"]
        angles = np.rad2deg(np.arccos(np.clip((ep["axis"] @ ref.T).max(1), -1, 1)))
        width = q[:, 7]
        opened = np.quantile(u[:, 7], 0.95)
        core = np.arange(g, r)[angles[g:r] >= angles[g:r].max() / 2]
        rows.append(
            dict(
                name=ep["name"],
                empty=float(np.median(width[: ep["opened"]])),
                held_min=float(width[core].min()),
                held_max=float(width[core].max()),
                open=float(opened),
                peak=float(angles[g:r].max()),
                open_angle=float(angles[u[:, 7] >= opened / 2].max()),
            )
        )

    def gap_mid(lower, upper):
        a, b = max(r[lower] for r in rows), min(r[upper] for r in rows)
        if a >= b:
            raise ValueError(f"Demonstrations do not separate {lower} and {upper}: {a}, {b}")
        return (a + b) / 2

    open_m = gap_mid("held_max", "open")
    waits = []
    for ep in episodes:
        g = ep["grasp"]
        issue = g + np.flatnonzero(ep["u"][g:, 7] >= open_m)[0]
        index = issue + np.flatnonzero(ep["q"][issue:, 7] >= open_m)[0]
        # One extra measured sampling interval accounts for discrete observation arrival.
        waits.append(float((ep["t"][index + 1] - ep["t"][issue]) * 1e-9))
    return dict(
        empty_m=gap_mid("empty", "held_min"),
        open_m=open_m,
        release_wait_s=max(waits),
        tilt_deg=gap_mid("open_angle", "peak"),
        joint_rotations=hand.rot.tolist(),
        hand_basis=hand.basis.tolist(),
        grasp_gravity_axes=ref.tolist(),
        fit=rows,
    )


def check(ep, profile, fault):
    q, u = ep["q"].copy(), ep["u"].copy()
    g, r = ep["grasp"], ep["release"]
    angles = np.rad2deg(
        np.arccos(np.clip((ep["axis"] @ np.array(profile["grasp_gravity_axes"]).T).max(1), -1, 1))
    )
    peak = g + angles[g:r].argmax()
    start = None
    if fault == "empty_grasp":
        start = g + 1
        q[start:r, 7] = 0
    elif fault == "tilted_open_request":
        start = peak
        u[peak, 7] = 0.1
    elif fault == "tilted_open_feedback":
        start = peak
        q[peak, 7] = 0.1
    elif fault == "stuck_at_release":
        start = r
        q[r:, 7] = np.median(q[g:r, 7])
    elif fault == "release_without_tilt":
        start = r
        q[g : r + 1, :7] = q[g, :7]
    guard = PouringGrip(profile)
    paused = None
    for i, (goal, measured) in enumerate(zip(u, q, strict=True)):
        try:
            guard.check(goal, measured, float(ep["t"][i] - ep["t"][0]) * 1e-9)
        except ValueError:
            paused = i
            break
    return dict(
        name=ep["name"],
        fault=fault,
        injection_row=None if start is None else int(start),
        stopped_at=paused,
        phase=guard.phase,
        reason=guard.fault,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("urdf", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    hand = NeroHand(args.urdf, "cpu")
    eps, _ = load_inventory(args.root / "aligned/inventory.json")
    train = [recorded(s, hand) for s in eps if s.split == "train"]
    profile = fit(train, hand)
    (args.root / "grip-profile.json").write_text(json.dumps(profile, indent=2))
    results = []
    for ep in (recorded(s, hand) for s in eps if s.split == "validation"):
        for fault in (
            "recorded",
            "empty_grasp",
            "tilted_open_request",
            "tilted_open_feedback",
            "stuck_at_release",
            "release_without_tilt",
        ):
            results.append(check(ep, profile, fault))
    (args.root / "grip-check.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print({k: profile[k] for k in ("empty_m", "open_m", "tilt_deg")})
    for fault in sorted({r["fault"] for r in results}):
        rows = [r for r in results if r["fault"] == fault]
        print(fault, [(r["stopped_at"], r["phase"]) for r in rows])


if __name__ == "__main__":
    main()
