"""Prepare real RAW stage frames and calibrate U2 to NERO from reader-marked training poses."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from owac.raw.all_bench_distillation import read_npz_member
from owac.runtime.bowl_postprocess import hand_center


def contact_rows(ep, split):
    opened = ep["q"][:, 7] > split
    events = np.flatnonzero(opened[1:] != opened[:-1]) + 1
    start = next(i for i, j in enumerate(events) if not opened[j])
    pairs = np.array(list(zip(events[start::2], events[start + 1 :: 2], strict=False)))
    distance = np.linalg.norm(ep["xyz"][pairs[:, 1], :2] - ep["xyz"][pairs[:, 0], :2], axis=1)
    return pairs[np.sort(np.argsort(distance)[-2:])].ravel()


def prepare(parent, old, out, marks):
    dest = out / "keyframes"
    dest.mkdir(parents=True, exist_ok=True)
    config = json.loads((old / "bowl-postprocess.json").read_text())
    marker_poses = []
    for path in sorted((old / "features").glob("bowl*.npz")):
        with np.load(path) as data:
            ep = {k: data[k] for k in ("q", "t", "xyz")}
        events = contact_rows(ep, config["width_split_m"])
        last = len(ep["q"]) - 1
        group, number = path.stem.split("-")
        raw = read_npz_member(parent / "aligned" / group / (number + ".npz"), "u2_raw.npy")
        idx = np.r_[0, events, np.minimum(last, events + 12), last]
        np.savez(dest / path.name, index=idx, u2=raw[idx], time=ep["t"][idx])

        def average(start, stop, raw=raw):
            return raw[start:stop].reshape(-1, 120, 4, 160, 4).mean((0, 2, 4), dtype=np.float32)

        # The original 15-frame mean used float64 accumulation, saved in float32.
        initial = raw[:15].reshape(-1, 120, 4, 160, 4).mean((0, 2, 4)).astype("f4")
        np.save(dest / (path.stem + "-initial.npy"), initial)
        np.save(
            dest / (path.stem + "-starts.npy"),
            np.stack([average(i, i + 15) for i in (0, 10, 20, 30)]),
        )
        np.save(
            dest / (path.stem + "-stack.npy"),
            np.stack(
                [
                    average(0, 15),
                    average(events[2] - 7, events[2] + 8),
                    average(last - 14, last + 1),
                ]
            ),
        )
        for frame, point in marks.get(f"{group}/{number}", {}).items():
            marker_poses.append((ep["q"][int(frame)], point))
    if len(marker_poses) < 10:
        raise ValueError("Mark the wrist bracket in at least ten varied training poses")
    joints, pixels = zip(*marker_poses, strict=True)
    calibrate(np.asarray(joints), np.asarray(pixels), config["geometry"], out)


def calibrate(q, pts, g, out):
    rows = list(range(len(q)))
    pos = hand_center(q, g)
    basis = np.stack(
        [hand_center(q, dict(g, tip=np.asarray(g["tip"]) + e)) - pos for e in np.eye(3)], -1
    )

    def camera(x):
        r = Rotation.from_rotvec(x[:3]).as_matrix()
        return (
            np.array([[np.exp(x[6]), 0, 79.5], [0, np.exp(x[6]), 59.5], [0, 0, 1]]) @ r
        ) @ np.c_[np.eye(3), -x[3:6]]

    def error(x):
        xyz = pos + basis @ x[7:]
        uv = np.c_[xyz, np.ones(len(q))] @ camera(x).T
        return uv[:, :2] / uv[:, 2:] - pts

    rot = Rotation.from_matrix([[0, 1, 0], [1, 0, 0], [0, 0, -1]]).as_rotvec()
    init = np.r_[rot, [-0.43, 0, 0.86], np.log(115), np.zeros(3)]
    fit = least_squares(lambda x: error(x).ravel(), init, loss="soft_l1")
    loo = []
    for i in range(len(q)):
        mask = np.arange(len(q)) != i
        x = least_squares(lambda x, mask=mask: error(x)[mask].ravel(), fit.x, loss="soft_l1").x
        loo.append(float(np.linalg.norm(error(x)[i])))
    result = dict(
        camera=camera(fit.x).tolist(),
        marker_link_offset_m=fit.x[7:].tolist(),
        rows=rows,
        points=pts.tolist(),
        quarter_native_px=np.linalg.norm(error(fit.x), axis=1).tolist(),
        leave_one_pose_out_px=loo,
    )
    (out / "calibration.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "old", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument(
        "--marks",
        type=Path,
        required=True,
        help="Training subset/episode -> frame index -> wrist bracket center at quarter resolution",
    )
    args = parser.parse_args()
    prepare(args.parent, args.old, args.out, json.loads(args.marks.read_text()))
