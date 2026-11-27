"""Choose a reachable pouring demonstration start and configure a separate field console."""

import argparse
import json
from pathlib import Path

import numpy as np

from owac.runtime.fruit_live import Travel


def reset_profile(starts, travel):
    episodes = sorted(starts)
    poses = np.asarray([starts[key]["start"] for key in episodes], dtype=np.float64)
    if poses.shape != (len(episodes), 8) or not len(episodes) or not np.isfinite(poses).all():
        raise ValueError("Expected finite measured seven-joint and aperture starts")
    bounded = np.clip(poses, travel.lower, travel.upper)
    costs = np.max(np.abs(bounded[:, None] - bounded[None, :]) / travel.step, axis=2)
    selected = int(np.argmin(costs.mean(axis=1)))
    goal = np.clip(
        bounded[selected],
        np.nextafter(travel.lower, travel.upper),
        np.nextafter(travel.upper, travel.lower),
    )
    key = episodes[selected]
    profile = dict(
        source=key + ", first paired pouring RAW and causal right-follower feedback",
        observation_ns=starts[key]["observation_ns"],
        joints_deg=np.rad2deg(goal[:7]).tolist(),
        gripper_mm=float(goal[7] * 1000),
    )
    return profile, dict(
        episode=key,
        starts=len(episodes),
        measured_rad_m=poses[selected].tolist(),
        goal_rad_m=goal.tolist(),
        clipped_axes=np.flatnonzero(bounded[selected] != poses[selected]).tolist(),
        mean_reset_ticks=float(costs[selected].mean()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("starts", "split", "catalog", "urdf", "departure", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    evidence = json.loads(args.starts.read_text())
    split = json.loads(args.split.read_text())
    train = {key for key, (part, _) in split["episodes"].items() if part == "train"}
    if (
        evidence["revision"] != split["revision"]
        or set(evidence["starts"]) != train
        or len(train) != 23
    ):
        raise ValueError("A complete training-only pouring split is required")
    if any(value["negative_width"] for value in evidence["starts"].values()):
        raise ValueError("Recorded aperture targets are not all physical widths")
    manifest = json.loads(args.catalog.read_text())
    ids = [item["id"] for item in manifest["candidates"]]
    if ids != ["owac", "dino-corp", "rawvla-pi05", "rawild-pi05"] or manifest["include_originals"]:
        raise ValueError("Expected only the four frozen pouring policies")
    manifest.update(
        task="Pouring 现场版",
        pouring_departures={"owac": "pouring-departure.json"},
        physical_gripper_width=True,
        record_policy_commands=True,
        command_frame_interval_s=0.5,
    )
    manifest["candidates"][0]["label"] += " · 反馈起步"
    travel = Travel(args.urdf, 0.2)
    profile, selection = reset_profile(evidence["starts"], travel)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("reset-pose", profile),
        ("pouring-departure", json.loads(args.departure.read_text())),
        ("field-catalog", manifest),
        ("reset-selection", selection),
    ):
        (args.out / (name + ".json")).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        )
    travel.load_reset_pose(args.out / "reset-pose.json")


if __name__ == "__main__":
    main()
