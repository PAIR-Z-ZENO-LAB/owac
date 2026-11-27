"""Propagate policy command deviations using a response fitted to pouring training telemetry."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from owac.raw.all_bench_distillation import load_inventory
from owac.runtime.fruit_live import Travel
from owac.runtime.pouring_grip import PouringGrip
from owac.training.fruit_grasp import NeroHand
from owac.training.pouring_feedback_data import opening_strokes, wrist_tilt
from owac.training.teleop_recovery import fixed_language, load_export, predict_cached
from owac.training.teleop_recovery_replay import RecoveryBank


def fit_response(episodes, travel):
    # Discrete first-order response on command changes with known availability;
    # delays searched from zero through one 12-command observation horizon.
    lags = np.arange(13) / 30
    xx = np.zeros((len(lags), 8))
    xy = np.zeros_like(xx)
    yy = np.zeros(8)
    dt = []
    count = 0
    for episode in episodes:
        with np.load(episode.path) as d:
            t, q = d["state_ns"], d["state"].astype("f8")
            ct, u = d["command_ns"], d["commands"].clip(travel.lower, travel.upper)
        use = (np.diff(t) > 0) & (t[:-1] - lags[-1] * 1e9 >= ct[0])
        times, a, b = t[:-1][use], q[:-1][use], q[1:][use]
        increments = b - a
        dt.extend(np.diff(t)[use] * 1e-9)
        count += len(a)
        yy += (increments**2).sum(0)
        for k, lag in enumerate(lags):
            j = np.searchsorted(ct, times - round(lag * 1e9), side="right") - 1
            delta = u[j] - a
            xx[k] += (delta**2).sum(0)
            xy[k] += (delta * increments).sum(0)
    gain = np.clip(xy / xx, 1e-6, 1 - 1e-6)
    residual = yy - 2 * gain * xy + gain**2 * xx
    servo = []
    for j, order in enumerate(residual.argmin(0)):
        servo.append(
            dict(
                tau_s=float(-np.median(dt) / np.log1p(-gain[order, j])),
                delay_s=float(lags[order]),
                train_increment_rmse=float(np.sqrt(residual[order, j] / count)),
                hold_increment_rmse=float(np.sqrt(yy[j] / count)),
            )
        )
    return dict(
        episodes=[ep.group_id for ep in episodes],
        observations=count,
        servo=servo,
        low=travel.lower.tolist(),
        high=travel.upper.tolist(),
    )


class PouringReplay(RecoveryBank):
    def __init__(self, episodes, root, cfg, tracking):
        visual = root / "rollout-visual"
        grids = []
        for src in episodes:
            with np.load(root / "feedback-cache" / (src.group_id + ".npz")) as z:
                path = visual / (src.group_id + ".npy")
                path.parent.mkdir(exist_ok=True, parents=True)
                if not path.is_file():
                    tmp = path.with_suffix(".tmp.npy")
                    np.save(tmp, z["visual"])
                    tmp.replace(path)
                grids.append(torch.from_numpy(z["grid"]).view(torch.bfloat16))
        super().__init__(episodes, visual, cfg, tracking)
        self.grid = torch.cat(grids).cuda()
        self.configure_output(cfg)
        self.history_rows(cfg["model"]["state_history_lags"])

    def predict(self, model, text, index, state, past=None):
        q = (state - self.mean) / self.sd
        with torch.autocast("cuda", dtype=torch.bfloat16):
            norm = predict_cached(model, self.visual[index], q, text)
            if hasattr(model, "feedback_head"):
                hist = (past - self.mean) / self.sd
                norm = model.correct(norm, self.grid[index], q, hist)
        return (norm.float() + 1) * self.half + self.zero + (state * self.relative)[:, None]


def score(bank, error, hand):
    q = bank.q.cpu()
    simulated = q + error.cpu()
    pos, rot = hand.pose(q)
    p, r = hand.pose(simulated)
    rows = []
    for i, (a, n) in enumerate(zip(bank.starts, bank.sizes, strict=True)):
        sl = slice(a, a + n)
        w = bank.target[sl, 0, 7].cpu().numpy()
        threshold = np.quantile(w, [0.1, 0.95]).mean()
        (opening, grasp), (release, stop) = opening_strokes(w, threshold)
        ref_tilt = wrist_tilt(rot[sl].numpy(), grasp)
        peak = grasp + ref_tilt[grasp:release].argmax()
        hold = np.arange(grasp, release)
        core = hold[ref_tilt[hold] >= ref_tilt[peak] / 2]
        sim_tilt = wrist_tilt(r[sl].numpy(), grasp)
        distance = np.linalg.norm((pos[sl] - p[sl]).numpy(), axis=-1) * 1000
        width = simulated[sl, 7].numpy()
        # Test a cycle locally around each recorded opening using midpoints between events.
        mid = (grasp + release) // 2
        opened = width[opening:grasp].max() > threshold
        closed = width[grasp:mid].min() < threshold
        released = width[mid:stop].max() > threshold
        rows.append(
            dict(
                name=bank.names[i],
                grasp_hand_mm=float(distance[grasp]),
                peak_hand_mm=float(distance[peak]),
                release_hand_mm=float(distance[release]),
                path_hand_mm=float(distance.mean()),
                tip_angle_error_deg=float(abs(sim_tilt[peak] - ref_tilt[peak])),
                tilt_rows=len(core),
                open_during_tilt=int((width[core] > threshold).sum()),
                paired_width_cycle=int(opened and closed and released),
            )
        )
    keys = (
        "grasp_hand_mm",
        "peak_hand_mm",
        "release_hand_mm",
        "path_hand_mm",
        "tip_angle_error_deg",
    )
    summary = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    for k in ("tilt_rows", "open_during_tilt", "paired_width_cycle"):
        summary[k] = sum(r[k] for r in rows)
    return dict(summary=summary, episodes=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "parent", "candidate", "urdf"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--field", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    episodes, _ = load_inventory(args.root / "aligned/inventory.json")
    train = [e for e in episodes if e.split == "train"]
    val = [e for e in episodes if e.split == "validation"]
    travel = Travel(args.urdf, 0.2)
    tracking_file = args.root / "servo-response.json"
    if args.field:
        tracking = json.loads(tracking_file.read_text())
    else:
        tracking = fit_response(train, travel)
        tracking_file.write_text(json.dumps(tracking, indent=2))
    candidate, _ = load_export(args.candidate)
    original, _ = load_export(args.parent)
    bank = PouringReplay(val, args.root, candidate.cfg, tracking)
    hand = NeroHand(args.urdf, "cpu")
    result = {}
    if args.field:
        profile = json.loads((args.root / "grip-profile.json").read_text())
        text = fixed_language(candidate)
        for gap in (0, 1, 3):
            bare = bank.rollout(candidate, text, 4, gaps=(gap,))
            result[f"raw_gap{gap}"] = score(bank, bare, hand)
            for guarded in (False, True):
                guards = [PouringGrip(profile) for _ in val]
                stops = {}
                last = bank.target[bank.start, 0].cpu().numpy().copy()

                def deliver(
                    plan,
                    row,
                    t,
                    frame,
                    measured,
                    guarded=guarded,
                    guards=guards,
                    stops=stops,
                    last=last,
                ):
                    command = plan[:, row].copy()
                    for i in range(len(val)):
                        if frame >= bank.sizes[i]:
                            continue
                        if guarded and i not in stops:
                            try:
                                command[i] = guards[i].command(
                                    command[i], measured[i], travel, bank.clocks[i][frame]
                                )
                            except ValueError as exc:
                                stops[i] = dict(
                                    name=val[i].group_id,
                                    row=frame,
                                    time_s=float(bank.clocks[i][frame]),
                                    reason=str(exc),
                                )
                                last[i, :7] = measured[i, :7]
                        if i in stops:
                            command[i] = last[i]
                        elif not guarded:
                            command[i, :7] = travel.command(command[i], last[i])[:7]
                    last[:] = command
                    return command

                errors = bank.rollout(candidate, text, 4, gaps=(gap,), postprocess=deliver)
                report = score(bank, errors, hand)
                report["guard_stops"] = list(stops.values())
                result[f"{'guarded' if guarded else 'limited'}_gap{gap}"] = report
        (args.root / "anchor/continuous.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False)
        )
        return
    for label, model, count in (
        ("original4", original, 4),
        ("original12", original, 12),
        ("candidate4", candidate, 4),
    ):
        for gap in (0, 1, 3):
            errors = bank.rollout(model, fixed_language(model), count, gaps=(gap,))
            report = score(bank, errors, hand)
            result[f"{label}_gap{gap}"] = report
            print(label, gap, report["summary"], flush=True)
    (args.root / "replay.json").write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
