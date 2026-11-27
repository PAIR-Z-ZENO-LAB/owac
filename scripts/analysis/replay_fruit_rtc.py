"""Replay fruit RTC and direct regression against the same held-out observations and clock."""

import argparse
import json
from pathlib import Path

import numpy as np

from owac.runtime.fruit_live import FEEDBACK_SECONDS, HZ, Observation, Prediction, Travel, row_at
from owac.runtime.fruit_rtc import NS, PrefixRegression, RegressionRTC


def paired_errors(errors):
    absolute = np.abs(errors)
    return dict(
        joint_mae_deg=float(np.rad2deg(absolute[:, :7]).mean()),
        gripper_mae_mm=float(absolute[:, 7].mean() * 1000),
    )


def known_prefix(traces, adapter, length):
    mask = traces["valid"].all(axis=1)
    raw = traces["predicted"][mask]
    target = traces["target"][mask]
    suffix = np.arange(length, raw.shape[1])
    corrected = np.stack(
        [
            adapter.condition(mu, suffix, np.arange(length), y[:length])
            for mu, y in zip(raw, target, strict=True)
        ]
    )
    return dict(
        complete_chunks=len(raw),
        known_commands=length,
        direct=paired_errors((raw[:, length:] - target[:, length:]).reshape(-1, 8)),
        conditioned=paired_errors((corrected - target[:, length:]).reshape(-1, 8)),
    )


def worker(times, delay):
    """One worker accepts the newest observation after its previous forward pass returns."""
    completed, frames = [], []
    index = 0
    available = times[0]
    while index < len(times):
        available = max(available, times[index])
        index = int(np.searchsorted(times, available, side="right") - 1)
        available += delay
        completed.append(available)
        frames.append(index)
        index += 1
    return np.array(completed), frames


def replay(traces, adapter, travel, delay, *, on_tick=None):
    baseline_errors, rtc_errors, direct_changes, rtc_changes, prefixes = [], [], [], [], []
    episodes = np.unique(traces["episode"])
    handoffs = stale_ticks = 0
    for episode in episodes:
        selected = traces["episode"] == episode
        t, states, predictions, labels, valid = (
            traces[name][selected] for name in ("time", "state", "predicted", "target", "valid")
        )
        ready, frames = worker(t, delay)
        rtc = RegressionRTC({"replay": adapter})
        rtc.start("replay", 0.0)
        previous, previous_slot = None, None
        for slot in range(int((t[-1] - ready[0]) * HZ) + 1):
            now = max(rtc.next_tick, (round(ready[0] * NS) + (slot * NS + HZ - 1) // HZ) / NS)
            result = int(np.searchsorted(ready, now, side="right") - 1)
            if result < 0:
                continue
            frame = frames[result]
            nearest_past = int(np.searchsorted(t, now, side="right") - 1)
            target_row = int((now - t[nearest_past]) * HZ + 0.5)
            p = Prediction(
                "replay", Observation(frame, t[frame], 0, (), None), predictions[frame], 0
            )
            if now - t[frame] > FEEDBACK_SECONDS:
                stale_ticks += 1
                rtc.start("replay", now)
                continue
            before = rtc.handoffs
            state = states[nearest_past]
            action = rtc.step(
                p, now, lambda raw, now=now, state=state: (now, travel.command(raw, state))
            )
            if rtc.handoffs > before:
                prefixes.append(rtc.prefix)
                handoffs += 1
            if (
                action is None
                or target_row >= valid.shape[1]
                or not valid[nearest_past, target_row]
            ):
                continue
            reference = labels[nearest_past, target_row]
            direct = travel.command(p.actions[row_at(p, now)], state)
            if on_tick is not None:
                on_tick(episode, now, state, reference, direct, action[-1])
            baseline_errors.append(direct - reference)
            rtc_errors.append(action[-1] - reference)
            if previous_slot == slot - 1:
                direct_changes.append(direct - previous[0])
                rtc_changes.append(action[-1] - previous[1])
            previous, previous_slot = (direct, action[-1]), slot
    return dict(
        episodes=len(episodes),
        delay_ms=delay * 1000,
        matched_control_ticks=len(rtc_errors),
        stale_ticks=stale_ticks,
        handoffs=handoffs,
        prefix_range=[min(prefixes), max(prefixes)],
        direct=paired_errors(baseline_errors),
        rtc=paired_errors(rtc_errors),
        direct_command_change=paired_errors(direct_changes),
        rtc_command_change=paired_errors(rtc_changes),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    args = parser.parse_args()
    travel = Travel(args.urdf, 0.2)
    for key in ("best", "final"):
        calibration = json.loads((args.directory / f"rtc-{key}.json").read_text())
        adapter = PrefixRegression(calibration["covariance"])
        with np.load(args.directory / f"val-{key}.npz") as f:
            traces = dict(f)
        report = dict(
            step=calibration["step"],
            frames=len(traces["time"]),
            oracle_prefix=[known_prefix(traces, adapter, k) for k in (1, 3)],
            replay=[replay(traces, adapter, travel, delay) for delay in (0.012, 0.080)],
        )
        print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
