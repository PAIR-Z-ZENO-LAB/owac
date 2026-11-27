"""Replay the deployed bowl processor on the paused field case and six held-out trajectories."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from owac.frameworks.raw_compact.temporal_spatial_regression import causal_history_indices
from owac.raw.all_bench_distillation import load_inventory
from owac.runtime.bowl_postprocess import BowlPostprocessor, hand_center
from owac.training.fruit_grasp import NeroHand
from owac.training.teleop_policy_eval import predict_actions
from owac.training.teleop_recovery import fixed_language, load_export
from owac.training.teleop_recovery_replay import RecoveryBank, score_replay


def field_case(root, profile, model, hand):
    records = [
        json.loads(s)
        for s in next((root / "field").glob("commands-*.jsonl")).read_text().splitlines()
    ]
    measured = np.array([r["feedback_rad_m"] for r in records])
    issued = np.array([r["issued_rad_m"] for r in records])
    raw = np.array([r["model_rad_m"] for r in records])
    mux = np.array([r["latest_mux_rad_m"] for r in records])
    t0 = records[0]["time_ns"]
    clock = (np.array([r["time_ns"] for r in records]) - t0) / 1e9
    proc = BowlPostprocessor(profile)
    for obs, action, timestamp in zip(measured, raw, clock, strict=True):
        proc.jaw(action[7], obs, None, timestamp)
    assert proc.phase == "holding"
    with np.load(root / "field/paused.npz") as snap:
        image, q, rows, wall = (snap[k] for k in ("raw", "state", "raw_rows", "wall"))
    t = (wall - t0) / 1e9
    ix = causal_history_indices(t, model.cfg["model"]["state_history_lags"], 1 / 30)
    native, scenes = predict_actions(model, image, q[rows], history=q[ix[rows]], return_scene=True)
    rows_result = []
    for i, row in enumerate(rows):
        # Each question starts from the same observed grasp and carry. No real release was issued.
        case = copy.deepcopy(proc)
        case.samples.clear()
        for j in range(row):
            case.jaw(native[i, 0, 7], q[j], None, t[j])
        width = case.jaw(native[i, 0, 7], q[row], scenes[i], t[row])
        rows_result.append(
            dict(
                model_mm=float(native[i, 0, 7] * 1000),
                issue_mm=float(width * 1000),
                score=case.score,
                phase=case.phase,
                capture_ns=int(wall[row]),
            )
        )
    reconstructed = hand_center(measured, profile["geometry"])
    reference = hand.pose(torch.tensor(measured[:, :7], dtype=torch.float32))[0].numpy()
    tail = np.flatnonzero(issued[:, 7] > 0)
    last_nonzero = int(tail[-1])
    # Mux values are the last one observed before sending the current command.
    possible = records[1:]
    exact = np.all(np.isclose(mux[1:], issued[:-1], rtol=0, atol=1e-7), axis=1)
    return dict(
        field_rows=len(records),
        duration_s=float(clock[-1]),
        last_nonzero=last_nonzero,
        zero_commands_at_end=len(records) - 1 - last_nonzero,
        closed_segment_s=float(clock[-1] - clock[last_nonzero + 1]),
        closed_model_range_mm=(raw[last_nonzero + 1 :, 7] * 1000)[
            [np.argmin(raw[last_nonzero + 1 :, 7]), np.argmax(raw[last_nonzero + 1 :, 7])]
        ].tolist(),
        closed_measured_range_mm=[
            float(measured[last_nonzero + 1 :, 7].min() * 1000),
            float(measured[last_nonzero + 1 :, 7].max() * 1000),
        ],
        next_mux_exact=int(exact.sum()),
        next_mux_possible=len(possible),
        max_observed_open_mm=float(measured[:, 7].max() * 1000),
        paused_feedback_mm=float(q[-1, 7] * 1000),
        independent_paused_pairs=rows_result,
        same_pose=np.max(np.linalg.norm(reference - reconstructed, axis=-1)).item(),
        real_new_commands=0,
    )


def missing_release(episodes, profile, schedules):
    """Suppress the native opening and replay recovery on measured states and images."""
    results = []
    for gaps in schedules:
        group = {}
        for name, ep in episodes.items():
            proc = BowlPostprocessor(profile)
            t = ep["t"]
            q = ep["q"]
            cmd = ep["command"][:, 7]
            split = profile["width_split_m"]
            is_open = cmd > split
            edges = np.flatnonzero(is_open[1:] != is_open[:-1]) + 1
            first_close = next(j for j in edges if not is_open[j])
            placed = next(j for j in edges if j > first_close and is_open[j])
            next_pick = next(j for j in edges if j > placed and not is_open[j])
            release = handoff = None
            chunk = capture = 0
            while capture < len(t):
                wait = gaps[chunk % len(gaps)]
                for j in range(capture + wait, min(capture + wait + 4, len(t))):
                    before = proc.phase
                    raw = 0.0 if j >= first_close else cmd[j]
                    width = proc.jaw(raw, q[j], ep["scene"][capture].astype("f4"), t[j])
                    if before == "holding" and proc.phase == "opening" and width > split:
                        release = j
                    if before == "after_release" and proc.phase == "native":
                        handoff = j
                capture += wait + 4
                chunk += 1
            group[name] = dict(
                recovered=release is not None,
                handed_back=handoff is not None,
                release_delta_s=float(t[release] - t[placed]) if release is not None else None,
                release_xyz_mm=float(1000 * np.linalg.norm(ep["xyz"][release] - ep["xyz"][placed]))
                if release is not None
                else None,
                handoff_delta_s=float(t[handoff] - t[next_pick]) if handoff is not None else None,
            )
        results.append(group)
    return results


def main(root, parent, urdf, profile_file):
    torch.set_num_threads(2)
    profile = json.loads(profile_file.read_text())
    model, _ = load_export(parent / "motion-history/policy-best.pt")
    text = fixed_language(model)
    hand = NeroHand(urdf, "cpu")
    field = field_case(root, profile, model, hand)
    (root / "field-release.json").write_text(json.dumps(field, indent=2))
    print(
        "field",
        [(r["score"], r["issue_mm"]) for r in field["independent_paused_pairs"]],
        flush=True,
    )
    val = [
        s for s in load_inventory(parent / "aligned/inventory.json")[0] if s.split == "validation"
    ]
    track = json.loads((parent / "field-tracking.json").read_text())
    bank = RecoveryBank(val, parent / "visual", model.cfg, track)
    bank.configure_output(model.cfg)
    bank.load_spatial(parent / "spatial")
    bank.history_rows(model.cfg["model"]["state_history_lags"])
    scene = []
    episodes = {}
    for n in bank.names:
        with np.load(root / "features" / (n.replace("/", "-") + ".npz")) as cache:
            scene.append(cache["scene"].astype("f4"))
            episodes[n] = dict(cache)
    previous = json.loads((parent / "execution-comparison.json").read_text())
    fault = missing_release(episodes, profile, previous["gaps"])
    (root / "missing-release.json").write_text(json.dumps(fault, indent=2))
    results = {"native": previous["results"]["history-4"]["protocols"]}
    for name in ("smoothed", "release", "combined"):
        runs = []
        for gap in previous["gaps"]:
            processors = [BowlPostprocessor(profile) for _ in bank.names]

            def process(raw, row, capture, frame, state, processors=processors, name=name):
                out = raw[:, row].copy()
                for i, proc in enumerate(processors):
                    if frame >= len(scene[i]):
                        continue
                    now = bank.clocks[i][frame]
                    if name != "release":
                        out[i, :7] = proc.arm.step(raw[i], state[i], now, row)
                    if name != "smoothed":
                        out[i, 7] = proc.jaw(out[i, 7], state[i], scene[i][capture], now)
                return out

            error = bank.rollout(model, text, 4, gaps=gap, postprocess=process)
            summ, eps = score_replay(bank, error, hand)
            runs.append(dict(summary=summ, episodes=eps))
        results[name] = runs
        print(
            name,
            [
                (
                    r["summary"]["events"],
                    round(r["summary"]["first_grasp_xy_mm"], 1),
                    round(r["summary"]["four_events_xyz_mm"], 1),
                )
                for r in runs
            ],
            flush=True,
        )
    (root / "feedback-replay.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("experiment", "parent", "urdf", "profile"):
        p.add_argument(k, type=Path)
    a = p.parse_args()
    main(a.experiment, a.parent, a.urdf, a.profile)
