"""Fit object-relative, phase-conditioned trajectories for two successive bowl transfers."""

import argparse
import json
from pathlib import Path

import numpy as np

from owac.runtime.bowl_localization import BowlLocator, exposure, rim_texture, texture


def unit_patch(feat, center, mask):
    r = len(mask) // 2
    x, y = np.rint(center).astype(int)
    patch = feat[y - r : y + r + 1, x - r : x + r + 1] * mask
    patch -= mask * (patch.sum() / mask.sum())
    return patch / max(np.linalg.norm(patch), 1e-8)


def fit_locator(train, images, first_frames, camera):
    from scipy.ndimage import gaussian_filter, map_coordinates

    # Approximate center clicks on training starts only, in the quarter-resolution U2 frame.
    # A radial rim fit refines each click; the deployed search scans the entire tabletop.
    first = {
        "bowl01": [
            [102, 50, 66, 54, 85, 75],
            [104, 53, 64, 54, 85, 75],
            [104, 53, 65, 54, 84, 76],
            [103, 54, 64, 54, 84, 76],
            None,
            [103, 53, 64, 55, 84, 76],
            [103, 54, 66, 55, 83, 77],
            [103, 53, 67, 54, 83, 77],
            [104, 53, 63, 55, 83, 76],
            None,
        ],
        "bowl02": [
            [105, 51, 67, 53, 88, 75],
            [105, 53, 64, 54, 88, 75],
            [104, 53, 64, 53, 88, 75],
            [105, 53, 65, 53, 88, 75],
            None,
            [104, 53, 64, 54, 87, 74],
            [104, 52, 65, 53, 88, 75],
            [104, 53, 65, 53, 88, 75],
            [105, 53, 65, 53, 88, 75],
            None,
        ],
        "bowl03": [
            [104, 51, 66, 53, 85, 75],
            [105, 51, 65, 54, 84, 75],
            [104, 51, 65, 53, 84, 75],
            [105, 51, 65, 54, 84, 76],
            None,
            [104, 51, 65, 53, 84, 76],
            [104, 51, 65, 54, 84, 76],
            [104, 51, 65, 53, 84, 76],
            [106, 50, 65, 54, 84, 76],
            None,
        ],
    }
    radius = 14
    cfg = dict(camera=camera, table_bounds=[36, 31, 140, 104], patch_radius=radius, banks=[])
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    mask = (x * x + y * y <= radius**2).astype("f4")
    theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    dx, dy = np.cos(theta), np.sin(theta)
    shifts = np.array(np.meshgrid(np.arange(-2, 2.01, 0.5), np.arange(-2, 2.01, 0.5))).reshape(
        2, -1
    )
    rr = np.arange(11.5, 14.1, 0.5)[None, :, None]
    for name in train:
        group, num = name.split("-")
        rough = np.asarray(first[group][int(num[-3:]) - 1]).reshape(3, 2)
        grid = images[name]
        im = gaussian_filter(grid, 0.7)
        feat = rim_texture(grid)
        centers = []
        for c in rough:
            xx = (c[0] + shifts[0])[:, None, None]
            yy = (c[1] + shifts[1])[:, None, None]

            def ring(r, im=im, xx=xx, yy=yy):
                return map_coordinates(im, [yy + r * dy, xx + r * dx], order=1)

            edge = abs(ring(rr + 0.8) - ring(rr - 0.8))
            strength = np.quantile(edge, 0.45, axis=-1).max(1)
            centers.append(c + shifts[:, np.argmax(strength)])
        cfg["banks"].append(
            dict(
                name=name,
                exposure=exposure(grid).tolist(),
                centers=np.array(centers).tolist(),
                templates=[unit_patch(feat, c, mask).tolist() for c in centers],
                min_score=[0.0, 0.0, 0.0],
            )
        )
    # Include the changing illumination in the first second before the bowls move.
    # Every appearance has the same measured center within its source recording.
    extra = []
    for item in cfg["banks"]:
        windows = first_frames[item["name"] + "-starts"]
        for window in windows[1:]:
            feat = rim_texture(window)
            extra.append(
                dict(
                    item,
                    exposure=exposure(window).tolist(),
                    templates=[unit_patch(feat, c, mask).tolist() for c in item["centers"]],
                )
            )
    cfg["banks"].extend(extra)
    # Each recording is located using other recordings. Learn a floor per exposure
    # from positives and the covered flower/white and the vacant black source at the end.
    evidence = {}
    for name in train:
        look = BowlLocator(dict(cfg, banks=[b for b in cfg["banks"] if b["name"] != name]))
        pos = np.min([look.locate(x)[1] for x in first_frames[name + "-starts"]], axis=0)
        single = look.locate(first_frames[name][0])[1]
        final = first_frames[name][-1]
        neg = look.locate(final)[1]
        look.bounds = [44, 38, 79, 69]
        neg[1] = look.locate(final, (1,))[1][0]
        evidence.setdefault(name[:6], []).append((np.minimum(pos, single), neg))
    for bank in cfg["banks"]:
        pairs = np.asarray(evidence[bank["name"][:6]])
        lo = pairs[:, 0].min(0)
        hi = pairs[:, 1].max(0)
        # Equal-cost separation when modes do not overlap; lower training quartile
        # of the class difference avoids relying on a single mislabeled absent response.
        bank["min_score"] = ((lo + np.minimum(hi, lo)) * 0.5).tolist()
    return cfg


def json_write(path, data):
    path.write_text(json.dumps(data, separators=(",", ":")) + "\n")


def fit_stack(out, train, locator, images):
    from scipy.ndimage import gaussian_filter, map_coordinates

    from owac.runtime.bowl_localization import StackAppearance

    r = locator.r
    y, x = np.mgrid[-r : r + 1, -r : r + 1]
    mask = (x * x + y * y <= r * r).astype("f4")
    shifts = np.array(np.meshgrid(np.arange(-4, 4.1, 0.5), np.arange(-4, 4.1, 0.5))).reshape(2, -1)
    theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    radius = np.arange(12, 15.1, 0.5)[None, :, None]
    templates = []
    exposures = []
    centers = []
    all_states = []
    for n in train:
        grid = np.load(out / "keyframes" / (n + "-stack.npy"))
        prior = locator.locate(images[n])[0][2]
        row = []
        points = []
        for label, im in enumerate(grid):
            cx = (prior[0] + shifts[0])[:, None, None]
            cy = (prior[1] + shifts[1])[:, None, None]
            smooth = gaussian_filter(im, 0.7)

            def rim(rad, smooth=smooth, cx=cx, cy=cy):
                return map_coordinates(
                    smooth, [cy + rad * np.sin(theta), cx + rad * np.cos(theta)], order=1
                )

            strength = np.quantile(abs(rim(radius + 0.8) - rim(radius - 0.8)), 0.45, axis=-1).max(1)
            c = prior + shifts[:, np.argmax(strength)] if label else prior
            row.append(unit_patch(texture(im), c, mask))
            points.append(c)
        centers.append(points)
        templates.append(row)
        exposures.append([exposure(g) for g in grid])
        all_states.append(grid)
    templates, exposures = np.array(templates), np.array(exposures)
    positives = []
    negatives = []
    correct = []
    for i, grid in enumerate(all_states):
        keep = np.arange(len(train)) != i
        model = StackAppearance(templates[keep], exposures[keep], [0] * 3, r)
        scores = np.array([model.read(g, centers[i][0])[2] for g in grid])
        positives.append(scores.diagonal())
        negatives.append(np.where(np.eye(3), -np.inf, scores).max(0))
        correct.append((scores.argmax(1) == np.arange(3)).tolist())
    lo = np.min(positives, 0)
    hi = np.max(negatives, 0)
    cutoff = (lo + np.minimum(hi, lo)) / 2
    print("stack train leave one out", np.sum(correct, 0), cutoff, flush=True)
    return dict(
        stack_templates=templates,
        stack_exposures=exposures,
        stack_cutoffs=cutoff,
        landed=np.array(centers)[:, 1:],
    ), correct


def distribution(bank):
    paths = bank["paths"]
    scenes = bank["scenes"]
    centers = bank["centers"]
    body = np.sqrt(np.mean((paths[:, None, :, 0] - paths[None, :, :, 0]) ** 2, -1))
    vision = np.sqrt(np.mean((scenes[:, None] - scenes[None, :]) ** 2, -1))
    xy = np.linalg.norm(centers[:, None] - centers[None, :], axis=-1)
    end = np.sqrt(np.mean((paths[:, None, :, -1] - paths[None, :, :, -1]) ** 2, -1))
    bank["central"] = np.median(end, axis=1)
    scales = [
        np.median(body, (0, 1)),
        np.median(vision, (0, 1)),
        np.r_[np.median(xy, (0, 1))[[0, 2, 1, 3]], 1.0],
        np.median(bank["central"], axis=0),
    ]
    bank["cost_scale"] = np.maximum(np.stack(scales, axis=1), 1.0e-6)
    return bank


def fit_policy(old, out, data, keys, images, train, loc):
    from itertools import product

    from scipy.ndimage import gaussian_filter1d

    from owac.runtime.bowl_flow import (
        adapt_contact,
        curve,
        pose_jacobian,
        sample_path,
        select_costs,
    )
    from owac.runtime.bowl_postprocess import hand_center

    cfg = json.loads((old / "bowl-postprocess.json").read_text())
    split = cfg["width_split_m"]
    geom = cfg["geometry"]
    limits = np.asarray(cfg["joint_limits_rad"])
    locator = BowlLocator(loc)
    tracks = []
    scenes = []
    centers = []
    periods = []
    reaches = []
    acks = []
    spans = []
    residuals = []
    stack, stack_cv = fit_stack(out, train, locator, images)
    archive = {}
    for name, e in data.items():
        q, u, t = e["q"], e["command"], e["t"]
        ix = keys[name]["index"][1:5]
        cuts = np.r_[0, ix, len(q) - 1]
        path = []
        starts = []
        span = []
        radii = []
        ack = []
        errors = []
        pts = locator.locate(images[name])[0]
        previous = locator.bounds
        x, y = pts[2].round().astype(int)
        r = locator.r
        locator.bounds = [x - r, y - r, x + r + 1, y + r + 1]
        stacked, score, cut = locator.locate(keys[name]["u2"][3], (0,))
        locator.bounds = previous
        flower = (
            stack["landed"][train.index(name), 0]
            if name in train
            else (stacked[0] if score[0] >= cut[0] else pts[2])
        )
        pts = np.vstack([pts, flower])
        edges = np.flatnonzero((u[1:, 7] > split) != (u[:-1, 7] > split)) + 1
        for s, (a, b) in enumerate(zip(cuts[:-1], cuts[1:], strict=True)):
            joints = q[a : b + 1, :7].astype(float)
            smooth = gaussian_filter1d(joints, 1.5, axis=0, mode="nearest")
            # Preserve contact orientation and height at both ends. One sample per equal joint arc.
            smooth[0], smooth[-1] = joints[0], joints[-1]
            length = np.r_[0, np.cumsum(np.linalg.norm(np.diff(smooth, axis=0), axis=1))]
            phase = np.linspace(0, length[-1], 65)
            path.append(np.column_stack([np.interp(phase, length, v) for v in smooth.T]))
            starts.append(e["scene"][a])
            err = np.abs(u[a : b + 1, :7] - q[a : b + 1, :7])
            span.append(np.quantile(err, 0.95, axis=0))
            errors.append(np.quantile(err[-max(4, int(0.3 * 30)) :], 0.95, axis=0))
            if s < 4:
                at = edges[np.argmin(abs(edges - b))]
                radii.append(np.linalg.norm(e["xyz"][at] - e["xyz"][b]))
                # Actual aperture a third of a second after the command stroke.
                after = np.searchsorted(t, t[b] + 0.333)
                ack.append(q[min(after, len(q) - 1), 7])
        archive[name] = dict(cuts=cuts, centers=pts, paths=np.array(path), scenes=np.array(starts))
        if name not in train:
            continue
        tracks.append(path)
        scenes.append(starts)
        centers.append(pts)
        periods.append(np.diff(t[cuts]))
        reaches.append(radii)
        acks.append(ack)
        spans.append(span)
        residuals.append(errors)
    names = np.array(train)
    arrays = dict(
        names=names,
        paths=np.array(tracks),
        scenes=np.array(scenes, dtype="f4"),
        centers=np.array(centers),
        landed=stack["landed"],
    )
    duration = np.median(periods, axis=0)
    plane = np.median([data[n]["xyz"][archive[n]["cuts"][1:5], 2] for n in train], axis=0)
    choices = np.array(
        [
            (body, vision, layout, central)
            for body, vision, layout, central in product(
                (0.5, 1.5), (0.25, 1.0, 3.0), (0.25, 1.0, 3.0), (0.5, 2.0, 5.0)
            )
        ]
    )
    losses = np.zeros((5, len(choices)))
    cache = {}
    for i, name in enumerate(train):
        fold = distribution({k: v[np.arange(len(train)) != i] for k, v in arrays.items()})
        row = archive[name]
        ep = data[name]
        for s in range(5):
            a, b = row["cuts"][s : s + 2]
            cost = select_costs(fold, s, ep["q"][a], row["scenes"][s], row["centers"])
            selected = cache[i, s - 1][0] if s in (1, 3) else np.argmin(cost @ choices.T, axis=0)
            by_expert = {}
            for j in np.unique(selected):
                ref = fold["paths"][j, s].copy()
                w = curve(np.linspace(0, 1, len(ref)))[:, None]
                ref += (1 - w) * (ep["q"][a, :7] - ref[0])
                if s < 4:
                    ref, _, _ = adapt_contact(
                        ref, w, s, fold, j, row["centers"], locator, plane, geom, limits
                    )
                true = row["paths"][s]
                xyz = hand_center(ref, geom)
                expert = hand_center(true, geom)
                pos_error = np.linalg.norm(xyz - expert, axis=-1)
                # Score wrist posture at the rim together with hand position.
                loss = (
                    pos_error.mean()
                    + pos_error[-1]
                    + 0.05 * np.sqrt(np.mean((ref[-1] - true[-1]) ** 2))
                )
                by_expert[j] = (float(loss), ref)
            losses[s] += np.array([by_expert[j][0] for j in selected])
            cache[i, s] = (selected, by_expert)
    best = losses.argmin(axis=1)
    for pick in (0, 2):
        best[pick : pick + 2] = (losses[pick] + losses[pick + 1]).argmin()
    gains = []
    all_width = np.concatenate([data[n]["command"][:, 7] for n in train])
    native_span = np.median(spans, axis=0)
    track_span = np.median(residuals, axis=0)
    for s, k in enumerate(best):
        num = den = 0.0
        for i, n in enumerate(train):
            selected, refs = cache[i, s]
            ref = refs[selected[k]][1]
            ep = data[n]
            a, b = archive[n]["cuts"][s : s + 2]
            q = ep["q"][a : b + 1, :7]
            u = ep["command"][a : b + 1, :7]
            frac = (ep["t"][a : b + 1] - ep["t"][a]) / (ep["t"][b] - ep["t"][a])
            guided = sample_path(ref, curve(frac))
            native = np.load(old / "features" / (n + "-plans.npy"), mmap_mode="r")[a : b + 1, 0, :7]
            delta = (
                np.clip(native - q, -native_span[s], native_span[s])
                * np.sin(np.pi * frac[:, None]) ** 2
            )
            num += np.sum((u - guided) * delta)
            den += np.sum(delta**2)
        gains.append(float(np.clip(num / max(den, 1.0e-9), 0, 1)))
    bank = distribution(arrays)
    bank.update(stack)
    bank.update(
        duration=duration,
        readout=choices[best],
        native_gain=gains,
        native_span=native_span,
        track_span=track_span,
    )
    np.savez_compressed(out / "trajectory-weights.npz", **bank)
    radii = np.quantile(reaches, 0.95, axis=0)
    ack = np.array(acks)
    config = {k: cfg[k] for k in ("geometry", "joint_limits_rad", "joint_penalty", "width_split_m")}
    config.update(
        kind="bowl_flow_v1",
        trajectory_weights="trajectory-weights.npz",
        locator=loc,
        plane_m=plane.tolist(),
        layout_frames=15,
        open_width_m=float(np.median(all_width[all_width > split])),
        closed_width_m=float(np.median(all_width[all_width < split])),
        contact_radius_m=np.r_[radii, radii.mean()].tolist(),
        ack_width_m=[float(np.quantile(ack[:, s], 0.95 if s % 2 == 0 else 0.05)) for s in range(4)],
    )
    stroke_frames = []
    for n in train:
        e = data[n]
        edges = np.flatnonzero(np.diff(e["command"][:, 7] > split)) + 1
        for at in archive[n]["cuts"][1:5]:
            stroke_frames.append(abs(at - edges[np.argmin(abs(edges - at))]))
    window = max(2, int(np.median(stroke_frames)) + 1)
    held = []
    for n in train:
        e = data[n]
        for q0, q1 in archive[n]["cuts"][1:5].reshape(2, 2):
            widths = e["q"][q0 + window : q1 - window, 7]
            if len(widths) > window:
                held.extend(
                    np.ptp(np.lib.stride_tricks.sliding_window_view(widths, window), axis=-1)
                )
    # On completed demonstrated stacks both visible rims share the destination footprint.
    # Allow one grid cell of quantization on the observed nesting offsets.
    config["stack_displacement_px"] = (
        1 + np.linalg.norm(stack["landed"] - arrays["centers"][:, None, 2], axis=-1).max(0)
    ).tolist()
    angles = []
    for n in train:
        e = data[n]
        edges = np.flatnonzero(np.diff(e["command"][:, 7] > split)) + 1
        at = archive[n]["cuts"][1:5]
        commanded = np.array([edges[np.argmin(abs(edges - i))] for i in at])
        ra = pose_jacobian(e["q"][at, :7], geom)[1]
        rb = pose_jacobian(e["q"][commanded, :7], geom)[1]
        angles.append(np.arccos(np.clip(((ra * rb).sum((1, 2)) - 1) / 2, -1, 1)))
    config["contact_angle_rad"] = np.quantile(angles, 0.95, axis=0).tolist()
    velocities = []
    for n in train:
        q = gaussian_filter1d(data[n]["q"][:, :7], 1.5, axis=0)
        velocities.append(abs(np.diff(q, axis=0)) / np.diff(data[n]["t"])[:, None])
    config["command_step_rad"] = (
        np.quantile(np.concatenate(velocities), 0.99, axis=0) / 30
    ).tolist()
    config["stroke_window"] = window
    config["held_width_variation_m"] = float(np.quantile(held, 0.95))
    json_write(out / "flow-policy.json", config)
    json_write(
        out / "fit.json",
        dict(
            training=train,
            validation=sorted(set(data) - set(train)),
            phase_durations_s=duration.tolist(),
            plane_m=plane.tolist(),
            radius_m=config["contact_radius_m"],
            ack_width_m=config["ack_width_m"],
            native_gain=gains,
            stack_leave_one_out=stack_cv,
            choices=choices.tolist(),
            loo_loss=(losses / len(train)).tolist(),
            selected=choices[best].tolist(),
        ),
    )
    print("trained", duration, radii, config["ack_width_m"], gains, flush=True)


def main(old, out):
    names = sorted(p.stem for p in (out / "keyframes").glob("bowl*.npz"))
    train = [n for n in names if not n.endswith(("005", "010"))]
    data = {n: dict(np.load(old / "features" / (n + ".npz"))) for n in names}
    images = {n: np.load(out / "keyframes" / (n + "-initial.npy")) for n in names}
    keys = {n: dict(np.load(out / "keyframes" / (n + ".npz"))) for n in names}
    camera = json.loads((out / "calibration.json").read_text())["camera"]
    starts = {n + "-starts": np.load(out / "keyframes" / (n + "-starts.npy")) for n in names}
    loc = fit_locator(train, images, {**{n: k["u2"] for n, k in keys.items()}, **starts}, camera)
    json_write(out / "locator.json", loc)
    locator = BowlLocator(loc)
    for n in names:
        centers, score, lo = locator.locate(images[n])
        one = locator.locate(keys[n]["u2"][0])[0]
        print(
            n,
            centers.round(1).ravel(),
            score.round(2),
            lo.round(2),
            "one frame shift",
            np.linalg.norm(centers - one, axis=1).round(1),
            flush=True,
        )
    for clip in sorted((out / "field").glob("*.npz")):
        with np.load(clip) as recorded:
            a = recorded["raw"][0, 0]
        print("field", locator.locate(a))
    fit_policy(old, out, data, keys, images, train, loc)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for a in ("old", "out"):
        p.add_argument(a, type=Path)
    args = p.parse_args()
    main(args.old, args.out)
