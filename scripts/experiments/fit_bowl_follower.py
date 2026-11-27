"""Fit the follower used for recorded-image bowl replays and compare the two field approaches."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar

from owac.training.fruit_grasp import NeroHand


def horizontal(hand, joints):
    positions = hand.pose(torch.as_tensor(joints[..., :7], dtype=torch.float32))[0]
    return positions.numpy()[..., :2] * 1000


def interpolate(clock, state, target):
    return np.stack([np.interp(target, clock, column) for column in state.T], axis=-1)


def model_errors(axis, delay, tau, segments):
    errors, stationary = [], []
    for clock, dt, state, following, command_time, actions in segments:
        index = np.searchsorted(command_time, clock + dt / 2 - delay, side="right") - 1
        target = actions[np.maximum(index, 0), axis]
        motion = (1 - np.exp(-dt / tau)) * (target - state[:, axis])
        change = following[:, axis] - state[:, axis]
        errors.append(change - motion)
        stationary.append(change)
    return np.mean(np.concatenate(errors) ** 2), np.mean(np.concatenate(stationary) ** 2)


def fit_response(segments):
    # Fit the two initial bouts, which share a follower. Hold out the second OWAC attempt.
    response = []
    for axis in range(8):
        candidates = []
        for delay in np.arange(0, 0.151, 0.005):
            fit = minimize_scalar(
                lambda tau, a=axis, d=delay: model_errors(a, d, tau, segments[:2])[0],
                bounds=(0.005, 0.6),
                method="bounded",
            )
            candidates.append((fit.fun, float(delay), float(fit.x)))
        _, delay, tau = min(candidates)
        error, stationary = model_errors(axis, delay, tau, segments[2:])
        response.append(
            dict(delay_s=delay, tau_s=tau, check_rmse=np.sqrt(error), hold_rmse=np.sqrt(stationary))
        )
    return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("diagnosis", "experiment", "urdf"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    scores = json.loads((args.diagnosis / "grasp-scores.json").read_text())
    limits = args.diagnosis.parent / "inference/motion-20261124/same-frame.json"
    low, high = np.asarray(json.loads(limits.read_text())["joint_limits_rad"]).T
    low, high = np.r_[low, 0], np.r_[high, 0.1]
    offsets = np.array([-3.0, -2.0, -1.5, -1.0, -0.5, 0.0])
    hand = NeroHand(args.urdf, "cpu")
    demos = []
    for record in scores["reference_coordinates"]:
        with np.load(args.experiment / "aligned" / (record["episode"] + ".npz")) as source:
            clock = (source["observation_ns"] - record["measured"]["time_ns"]) / 1e9
            demos.append(horizontal(hand, interpolate(clock, source["state"], offsets)))
    references = np.stack(demos)
    (trace,) = (args.diagnosis / "field").glob("commands-*.jsonl")
    rows = map(json.loads, trace.read_text().splitlines())
    bouts = itertools.groupby(rows, key=lambda row: row["policy"])
    segments, approaches = [], []
    for (name, source), score in zip(bouts, scores["field"], strict=True):
        records = list(source)
        if name != score["policy"] or records[0]["time_ns"] != score["start_ns"]:
            raise ValueError("Field scores and command trace refer to different motion bouts")
        first = records[0]["time_ns"]
        clock = (np.array([r["feedback_ns"] for r in records]) - first) / 1e9
        command_time = (np.array([r["time_ns"] for r in records]) - first) / 1e9
        q = np.array([r["feedback_rad_m"] for r in records])
        actions = np.clip([r["issued_rad_m"] for r in records], low, high)
        if name == "owac":
            xy = horizontal(hand, interpolate(clock, q, offsets + score["closure_feedback_s"]))
            approaches.append(
                dict(
                    offsets_seconds=offsets.tolist(),
                    xy_mm=xy.tolist(),
                    nearest_demo_xy_mm=np.linalg.norm(references - xy, axis=-1).min(0).tolist(),
                    y_above_demo_range_mm=(xy[:, 1] - references[:, :, 1].max(0)).tolist(),
                )
            )
        dt = np.diff(clock)
        # Field starts include controller bring-up. Match consecutive fresh feedback packets.
        valid = (dt > 0) & (clock[:-1] > 0.2)
        segments.append(
            (clock[:-1][valid], dt[valid], q[:-1][valid], q[1:][valid], command_time, actions)
        )
    result = dict(
        phase_aligned_field=approaches,
        servo=fit_response(segments),
        low=low.tolist(),
        high=high.tolist(),
        fit=(
            "First OWAC and CoRP control traces. One-step prediction checked on the "
            "second OWAC trace. Used only to propagate command deviations relative to "
            "observed expert dynamics in offline replay."
        ),
    )
    path = args.output or args.experiment / "field-tracking.json"
    path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
