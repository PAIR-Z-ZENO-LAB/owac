"""Learn first-release and next-grasp cues from the 24 development-training demonstrations."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from owac.runtime.bowl_postprocess import motion_features


def read_features(root):
    features = {}
    for path in sorted((root / "features").glob("bowl*.npz")):
        with np.load(path) as archive:
            features[path.stem] = {k: archive[k].astype("f4") for k in archive.files}
    return features


def transitions(values, split):
    opened = values > split
    edges = np.flatnonzero(opened[:-1] != opened[1:]) + 1
    first = next(i for i, row in enumerate(edges) if not opened[row])
    return edges[first:]


def event_data(episodes, split, event):
    data = {}
    index = 0 if event == "release" else 1
    for name, ep in episodes.items():
        measured = transitions(ep["q"][:, 7], split)
        # Pair each pickup with its release. One bright demonstration regrips near the source
        # before transporting the bowl. Retain its two actual transfers in chronological order.
        pairs = np.array(list(zip(measured[::2], measured[1::2], strict=False)))
        distance = np.linalg.norm(ep["xyz"][pairs[:, 1], :2] - ep["xyz"][pairs[:, 0], :2], axis=1)
        measured = pairs[np.sort(np.argsort(distance)[-2:])].ravel()
        command = transitions(ep["command"][:, 7], split)
        start, end = measured[index : index + 2]
        candidates = command[(index + 1) % 2 :: 2]
        target = min(candidates, key=lambda i: abs(i - end))
        stop = measured[index + 2] if len(measured) > index + 2 else len(ep["q"])
        ix = np.arange(start, stop)
        xyz = ep["xyz"][ix]
        past = np.maximum(np.searchsorted(ep["t"], ep["t"][ix] - 4 / 30, side="right") - 1, 0)
        velocity = (xyz - ep["xyz"][past]) / (ep["t"][ix] - ep["t"][past])[:, None]
        geo = motion_features(xyz, xyz[0], np.maximum.accumulate(xyz[:, 2]), velocity)
        # At the start of a 4-action plan, a pending expert transition is already actionable.
        # End positive training at measured completion, before the next motion begins.
        fit = ix <= max(end, target)
        label = ix >= target - 3
        data[name] = dict(
            x=geo,
            scene=ep["scene"][ix],
            y=label,
            fit=fit,
            time=ep["t"][ix],
            xyz=xyz,
            target=target - start,
        )
    return data


def fit(data, names, rank):
    geo = np.concatenate([data[n]["x"][data[n]["fit"]] for n in names])
    scene = np.concatenate([data[n]["scene"][data[n]["fit"]] for n in names])
    y = np.concatenate([data[n]["y"][data[n]["fit"]] for n in names])
    mean = geo.mean(0)
    sd = geo.std(0)
    z = (geo - mean) / sd
    center = scene.mean(0)
    vision = np.empty((scene.shape[1], 0))
    if rank:
        # All projections are fitted inside the training fold.
        centered = scene - center
        eig, axes = np.linalg.eigh(centered.T @ centered / len(centered))
        vision = axes[:, -rank:] / np.sqrt(eig[-rank:])
        z = np.column_stack((z, centered @ vision))
    z = np.column_stack((z, np.ones(len(z))))
    # Balance the short release stroke and the longer transport segment per episode.
    weights = []
    for name in names:
        labels = data[name]["y"][data[name]["fit"]]
        weights.extend(np.where(labels, 0.5 / labels.sum(), 0.5 / (~labels).sum()))
    weights = np.asarray(weights) / len(names)

    def loss(theta):
        pred = z @ theta
        reg = theta.copy()
        reg[-1] = 0
        value = weights @ (np.logaddexp(0, pred) - y * pred) + 0.01 * (reg @ reg) / 2
        grad = ((expit(pred) - y) * weights) @ z + 0.01 * reg
        return value, grad

    sol = minimize(loss, np.zeros(z.shape[1]), jac=True, method="L-BFGS-B")
    if not sol.success:
        raise RuntimeError(sol.message)
    motion = sol.x[: geo.shape[1]] / sd
    vis = vision @ sol.x[geo.shape[1] : -1]
    bias = sol.x[-1] - mean @ motion - center @ vis
    return dict(motion=motion.tolist(), scene=vis.tolist(), bias=float(bias))


def score_trace(row, model):
    return row["x"] @ model["motion"] + row["scene"] @ model["scene"] + model["bias"]


def event_error(scores, row, threshold):
    crossed = np.flatnonzero(scores >= threshold)
    at = int(crossed[0]) if len(crossed) else len(scores) - 1
    target = row["target"]
    return dict(
        found=bool(len(crossed)),
        dt=float(row["time"][at] - row["time"][target]),
        distance_mm=float(1000 * np.linalg.norm(row["xyz"][at] - row["xyz"][target])),
    )


def select(data, train):
    choices = []
    for rank in (0, 8, 24, 48):
        scores = {}
        for fold in range(4):
            # Cycle within each lighting, leaving two entire episodes per lighting out of each fold.
            valid = train[fold::4]
            model = fit(data, [n for n in train if n not in valid], rank)
            scores.update({n: score_trace(data[n], model) for n in valid})
        for threshold in np.linspace(-1.5, 3, 19):
            metrics = [event_error(scores[n], data[n], threshold) for n in train]
            choices.append(
                dict(
                    rank=rank,
                    threshold=float(threshold),
                    found=sum(r["found"] for r in metrics),
                    mae_s=float(np.mean([abs(r["dt"]) for r in metrics])),
                    mean_distance_mm=float(np.mean([r["distance_mm"] for r in metrics])),
                )
            )
    # Retain a visual cue for actuation. Rank zero is the geometry-only ablation.
    best = min(
        (r for r in choices if r["rank"]),
        key=lambda r: (-r["found"], r["mean_distance_mm"], r["mae_s"]),
    )
    model = fit(data, train, best["rank"])
    model["bias"] -= best["threshold"]
    return model, best, choices


def main(root):
    ep = read_features(root)
    train = sorted(n for n in ep if not n.endswith(("005", "010")))
    valid = sorted(set(ep) - set(train))
    widths = np.concatenate([ep[n]["command"][:, 7] for n in train])
    # Two-means on demonstration setpoints estimates closed/open modes and their separation.
    centers = np.quantile(widths, [0.1, 0.9])
    for _ in range(100):
        mask = widths > centers.mean()
        new = np.array([widths[~mask].mean(), widths[mask].mean()])
        if np.array_equal(new, centers):
            break
        centers = new
    split = float(centers.mean())
    profile = dict(
        geometry=json.loads((root / "geometry.json").read_text()),
        width_split_m=split,
        motion_window_s=4 / 30,
        open_width_m=float(np.median(widths[widths > split])),
        events={},
    )
    results = dict(training=train, validation=valid, events={})
    for event in ("release", "next_grasp"):
        data = event_data(ep, split, event)
        model, best, choices = select(data, train)
        profile["events"][event] = model
        rows = {n: event_error(score_trace(data[n], model), data[n], 0) for n in valid}
        results["events"][event] = dict(selected=best, validation=rows, candidates=choices)
        print(event, best, rows, flush=True)
    (root / "release-fit.json").write_text(json.dumps(results, indent=2))
    (root / "release-profile.json").write_text(json.dumps(profile, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    main(parser.parse_args().experiment)
