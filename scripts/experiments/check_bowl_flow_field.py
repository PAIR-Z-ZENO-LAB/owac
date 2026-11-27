"""Diagnose the paused rollout and measure responses to individual bowl moves."""

import argparse
import json
from pathlib import Path

import numpy as np

from owac.runtime.bowl_flow import BowlFlow, pose_jacobian, translate_path
from owac.runtime.bowl_localization import raw_grid
from owac.runtime.bowl_postprocess import hand_center


def field(out, proc, center):
    log = next((out / "field").glob("commands-*.jsonl"))
    rows = list(map(json.loads, log.read_text().splitlines()))
    q = np.array([r["feedback_rad_m"] for r in rows])
    u = np.array([r["issued_rad_m"] for r in rows])
    raw = np.array([r["model_rad_m"] for r in rows])
    mux = np.array([r["latest_mux_rad_m"] for r in rows])
    t = (np.array([r["time_ns"] for r in rows]) - rows[0]["time_ns"]) / 1.0e9
    stage = np.array([r["postprocess_phase"] for r in rows])
    opening = np.flatnonzero(stage == "opening")[0]
    confirm = np.flatnonzero(stage == "after_release")[0]
    xyz = hand_center(q, proc.geo)
    xy = np.linalg.norm(xyz[: opening + 1, :2] - xyz[opening, :2], axis=1)
    outer = proc.profile["contact_radius_m"][1]
    first = np.flatnonzero(xy > outer)[-1] + 1
    paused = np.load(out / "field/black-paused.npz")
    mean = np.mean([raw_grid(x[0]) for x in paused["raw"]], axis=0)
    points, vals, cutoffs = proc.locator.locate(mean)
    label, stack, stack_scores = proc.stack.read(
        mean, center, proc.profile["stack_displacement_px"][0]
    )
    # Log the isolated endpoint lookup as weak evidence when the gripper overlaps the black rim.
    black_end = points[1]
    tail = t >= t[-1] - 25
    return dict(
        source=log.name,
        rows=len(t),
        duration_s=t[-1],
        logged_phase_sequence=stage[np.r_[0, np.flatnonzero(stage[1:] != stage[:-1]) + 1]].tolist(),
        at_first_release=float(t[opening]),
        arrived_xy_first_release_s=float(t[first]),
        approach_to_release_s=float(t[opening] - t[first]),
        xy_radius_m=outer,
        driver_open_s=float(t[confirm] - t[opening]),
        second_closed_commands=int((u[confirm:, 7] < proc.profile["width_split_m"]).sum()),
        second_interval_s=float(t[-1] - t[confirm]),
        last25s_rows=int(tail.sum()),
        last25s_width_mm=(1000 * np.quantile(raw[tail, 7], [0, 0.5, 1])).tolist(),
        observed_stalled_xyz_mm=(1000 * xyz[-1]).tolist(),
        last_scene_feedback_mm=float(1000 * paused["state"][-1, 7]),
        mux_next_matching=int(
            np.all(np.isclose(mux[1:], u[:-1], rtol=0, atol=1.0e-7), axis=1).sum()
        ),
        mux_possible=len(t) - 1,
        stalled_detection_xy=black_end.tolist(),
        start_margins=(vals - cutoffs).tolist(),
        paused_stack=["white", "flower", "black"][label] if label >= 0 else "unconfirmed",
        stack_xy=stack[label].tolist() if label >= 0 else None,
        paused_stack_scores=stack_scores,
        paused_stack_roi=(
            f"Center marked in paused U2, quarter grid {tuple(center)}. "
            "Deployment uses the initial white rim."
        ),
        mechanical_trials_new=0,
    )


def shift_bowl(image, center, delta):
    """Translate one RAW bowl footprint and fill the vacated sliver from its surround."""
    yy, xx = np.indices(image.shape)
    d = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    radius = 14
    mask = d <= radius**2
    ring = (d > radius**2) & (d <= (radius + 2) ** 2)
    shifted = np.roll(mask, (delta[1], delta[0]), axis=(0, 1))
    out = image.copy()
    out[mask & ~shifted] = np.median(image[ring])
    out[shifted] = np.roll(image, (delta[1], delta[0]), axis=(0, 1))[shifted]
    return out


def check(out, old, center):
    p = BowlFlow.load(out / "flow-policy.json")
    record = field(out, p, center)
    (out / "field-diagnosis.json").write_text(json.dumps(record, indent=2))
    shifts = []
    ownership = []
    frames = []
    goals = []
    for name in json.loads((out / "fit.json").read_text())["validation"]:
        views = np.load(out / "keyframes" / (name + "-stack.npy"))
        centers = p.locator.locate(views[0])[0]
        stages = [
            p.stack.read(i, centers[2], p.profile["stack_displacement_px"][1])[0] for i in views
        ]
        frames.append(dict(name=name, labels=stages))
        for obj, stage in enumerate((0, 2, 1)):
            for change in ((3, 0), (-3, 0), (0, 3), (0, -3)):
                image = shift_bowl(views[0], centers[obj], change)
                xy, scores, limits = p.locator.locate(image)
                accepted = bool(
                    np.all(scores >= limits)
                    and min(np.linalg.norm(xy[i] - xy[j]) for i, j in ((0, 1), (0, 2), (1, 2)))
                    > 2 * (p.locator.r - 2)
                )
                diff = xy[obj] - centers[obj] - change
                level = p.profile["plane_m"][stage]
                actual = p.locator.table_xy(xy[obj], level) - p.locator.table_xy(
                    centers[obj], level
                )
                expected = p.locator.table_xy(centers[obj] + change, level) - p.locator.table_xy(
                    centers[obj], level
                )
                shifts.append(
                    dict(
                        name=name,
                        obj=obj,
                        requested_grid_px=list(change),
                        error_grid_px=float(np.linalg.norm(diff)),
                        accepted=accepted,
                        requested_mm=float(1000 * np.linalg.norm(expected)),
                        error_mm=float(1000 * np.linalg.norm(actual - expected)),
                    )
                )
        with np.load(old / "features" / (name + ".npz")) as d:
            cut = np.load(out / "keyframes" / (name + ".npz"))["index"][1:5]
            raw_q = d["q"][cut, :7]
        # Measure transport in both axes at all four held-out contact orientations.
        for off in ((0.025, 0, 0), (-0.025, 0, 0), (0, 0.025, 0), (0, -0.025, 0)):
            target = translate_path(raw_q, np.broadcast_to(off, (4, 3)), p.geo, p.limits)
            x, r, _ = pose_jacobian(raw_q, p.geo)
            xx, rr, _ = pose_jacobian(target, p.geo)
            goals.extend((1000 * np.linalg.norm(xx - x - off, axis=-1)).tolist())
            ownership.extend(
                np.rad2deg(np.arccos(np.clip(((r * rr).sum((1, 2)) - 1) / 2, -1, 1))).tolist()
            )
    develop = json.loads((out / "feedback-flow.json").read_text())["results"]
    score = {}
    for k, runs in develop.items():
        rows = [v for run in runs for v in run["episodes"].values()]
        ev = np.array([[r["xyz_error_mm"] for r in v["width_events"]] for v in rows])
        score[k] = dict(
            cases=len(rows),
            four_events=int(sum(len(v["width_events"]) == 4 for v in rows)),
            retired=int(sum(v["end"] == "finished" for v in rows)),
            stage_mean_mm=ev.mean(0).tolist(),
            stage_max_mm=ev.max(0).tolist(),
            finish_s=[v["finish_s"] for v in rows],
            recorded_black_final=sum(v.get("stack_verdict") == "black" for v in rows),
            recorded_flower_mid=sum(v.get("flower_observed") is True for v in rows),
        )
    geom = dict(
        moves=shifts,
        approved=sum(x["accepted"] for x in shifts),
        median_shift_mm=float(np.median([x["requested_mm"] for x in shifts])),
        median_detect_error_mm=float(np.median([x["error_mm"] for x in shifts if x["accepted"]])),
        p95_detect_error_mm=float(
            np.quantile([x["error_mm"] for x in shifts if x["accepted"]], 0.95)
        ),
        independent_stack_frames=frames,
        ik_25mm_cases=len(goals),
        ik_max_error_mm=max(goals),
        ik_max_angle_deg=max(ownership),
        replays=score,
    )
    (out / "verification.json").write_text(json.dumps(geom, indent=2))
    print(json.dumps(record, ensure_ascii=False))
    print({k: v for k, v in geom.items() if k not in ("moves", "replays")})
    print(score)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("old", type=Path)
    p.add_argument(
        "--center",
        type=float,
        nargs=2,
        required=True,
        help="Marked stack center at quarter resolution",
    )
    a = p.parse_args()
    check(a.out, a.old, a.center)
