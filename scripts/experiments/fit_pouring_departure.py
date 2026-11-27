"""Extract the first moving action chunk shared by the pouring training demonstrations."""

import argparse
import json
from pathlib import Path

import numpy as np

from owac.raw.all_bench_distillation import load_inventory
from owac.runtime.fruit_live import HZ, Travel


def fit(episodes, travel, horizon, open_m):
    candidates = []
    starts = []
    for episode in episodes:
        if episode.split != "train":
            continue
        with np.load(episode.path) as data:
            state, time = data["state"], data["state_ns"]
            command, clock = data["commands"], data["command_ns"]
        starts.append(state[0].copy())
        # J2 starts at the lower stop. Take one model horizon from its first
        # measured inward crossing, before the demonstrator opens to approach the cup.
        start = np.flatnonzero(state[:, 1] > travel.lower[1])[0]
        stop = np.searchsorted(time, time[start] + round(horizon * 1e9 / HZ))
        section = slice(start, stop + 1)
        indices = np.searchsorted(clock, time[section], side="right") - 1
        if np.any(indices < 0) or stop >= len(time):
            raise ValueError(f"Missing departure commands in {episode.group_id}")
        goals = np.clip(command[indices], travel.lower, travel.upper)
        opening = np.flatnonzero(command[:, 7] >= open_m)[0]
        if time[stop] >= clock[opening]:
            raise ValueError(f"Departure overlaps cup approach in {episode.group_id}")
        shoulder = state[section, 1]
        increasing = np.r_[True, shoulder[1:] > np.maximum.accumulate(shoulder)[:-1]]
        if np.any(goals[increasing, 1] <= shoulder[increasing]) or increasing.sum() < 2:
            raise ValueError(f"Departure lacks inward targets in {episode.group_id}")
        clock_grid = time[start] + np.arange(horizon + 1) * 1e9 / HZ
        points = np.stack(
            [np.interp(clock_grid, time[section], state[section, axis]) for axis in range(7)],
            axis=1,
        )
        onset = np.flatnonzero(command[: indices[0] + 1, 1] <= travel.lower[1])
        first = int(onset[-1] + 1) if len(onset) else 0
        seconds = (time[stop] - clock[first]) / 1e9
        candidates.append(
            (episode.group_id, shoulder[increasing], goals[increasing], points, seconds)
        )
    traces = np.stack([row[3] for row in candidates]) / travel.step[:7]
    distances = np.linalg.norm(traces[:, None] - traces[None], axis=-1).mean((1, 2))
    medoid = int(distances.argmin())
    name, shoulder, goals, _, _ = candidates[medoid]
    return dict(
        episode=name,
        reference_start=starts[medoid].tolist(),
        start_lower=np.min(starts, axis=0).tolist(),
        start_upper=np.max(starts, axis=0).tolist(),
        train_episodes=[c[0] for c in candidates],
        horizon=horizon,
        shoulder_rad=shoulder.tolist(),
        targets_rad_m=goals.tolist(),
        longest_departure_s=max(c[4] for c in candidates),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("inventory", "urdf", "grip", "output"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--horizon", type=int, required=True)
    args = parser.parse_args()
    episodes, _ = load_inventory(args.inventory)
    profile = fit(
        episodes, Travel(args.urdf, 0.2), args.horizon, json.loads(args.grip.read_text())["open_m"]
    )
    args.output.write_text(json.dumps(profile, indent=2) + "\n")


if __name__ == "__main__":
    main()
