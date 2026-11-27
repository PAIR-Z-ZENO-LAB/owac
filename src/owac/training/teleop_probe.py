"""Frozen, RAW-only spatial evaluation on held-out physical fruit trajectories."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from owac.raw.all_bench_distillation import read_npz_member
from owac.raw.teleop_mcap import CAMERAS

# Fixed regions from the collection's U2 illumination audit, in native pixel coordinates.
WALL = (142, 22, 208, 64)
MAT = (540, 236, 600, 280)


def brightness_pairs(raw, count):
    x0, y0, x1, y1 = WALL
    area = raw[:, y0:y1, x0:x1]
    green = np.concatenate((area[:, 0::2, 1::2], area[:, 1::2, 0::2]), axis=2)
    level = np.median(green, axis=(1, 2))
    order = np.argsort(level, kind="stable")
    # Disjoint percentiles even in a nearly constant-light episode. Ties use time order.
    quantiles = np.linspace(0.05, 0.20, count)
    low = order[np.rint((len(order) - 1) * quantiles).astype(int)]
    high = order[np.rint((len(order) - 1) * (1 - quantiles)).astype(int)]
    return level, list(zip(low.tolist(), high.tolist(), strict=True))


def match_patches(reference, query, grid, patch, regions=None):
    """Query starts one native patch lower and to the right of the reference."""
    height, width = grid
    yy, xx = torch.meshgrid(
        torch.arange(height, device=query.device),
        torch.arange(width, device=query.device),
        indexing="ij",
    )
    valid = (yy + 1 < height) & (xx + 1 < width)
    if regions:
        keep = torch.zeros_like(valid)
        for x0, y0, x1, y1 in regions:
            keep |= (
                ((xx + 1) * patch >= x0)
                & ((xx + 2) * patch <= x1)
                & ((yy + 1) * patch >= y0)
                & ((yy + 2) * patch <= y1)
            )
        valid &= keep
    r = F.normalize(reference.float() - reference.float().mean(1, keepdim=True), dim=-1)
    q = F.normalize(query.float() - query.float().mean(1, keepdim=True), dim=-1)
    candidate = (q[:, valid.flatten()] @ r.transpose(1, 2)).argmax(-1)
    y, x = yy[valid] + 1, xx[valid] + 1
    hit = candidate == (y * width + x)
    distance = (
        (candidate // width - y).float().square() + (candidate % width - x).float().square()
    ).sqrt() * patch
    return np.array([int(hit.sum()), hit.numel(), float(distance.sum())], np.float64)


def preview(raw, black, white):
    """Display only: fixed BGGR cell values, fixed white balance and sRGB transfer."""
    a = raw.astype(np.float32)
    rgb = np.stack((a[1::2, 1::2], (a[0::2, 1::2] + a[1::2, 0::2]) / 2, a[0::2, 0::2]), axis=-1)
    rgb = np.clip((rgb - black) / (white - black) * np.array([2.5, 1, 12 / 7]), 0, 1)
    srgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)
    return np.rint(255 * srgb).astype(np.uint8)


@torch.inference_mode()
def evaluate_teleop(encoder, training, validation, cfg, root, step, device):
    encoder.eval()
    folder = Path(root) / "visual-probe"
    folder.mkdir(exist_ok=True)
    p = cfg["model"]["patch_size"]
    sensor = cfg["model"]["measurement_encoder"]["sensor"]
    black, white = sensor["black_level"], sensor["white_level"]

    def tokens(images):
        x = torch.as_tensor(np.ascontiguousarray(images), device=device).float()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            return encoder(x)[1].float()

    def camera_arrays(source):
        return [read_npz_member(source.path, k + ".npy") for k in CAMERAS]

    evaluation = cfg["evaluation"]
    basis_file = Path(evaluation.get("pca_basis", folder / "train-initial-pca.npz"))
    if step == evaluation.get("baseline_step", 0) and "pca_basis" not in evaluation:
        features = []
        for source in training:
            views = camera_arrays(source)
            data = np.stack([v[len(v) // 2] for v in views])
            feats = tokens(data)
            features.append(feats[:, ::4].flatten(0, 1).cpu().numpy())
        X = np.concatenate(features).astype(np.float64)
        mean = X.mean(0)
        X -= mean
        _, V = np.linalg.eigh(X.T @ X)
        basis = V[:, -3:][:, ::-1].copy()
        for axis in basis.T:
            if axis[np.abs(axis).argmax()] < 0:
                axis *= -1
        bounds = np.quantile(X @ basis, [0.01, 0.99], axis=0)
        np.savez(
            basis_file, mean=mean.astype("f4"), basis=basis.astype("f4"), bounds=bounds.astype("f4")
        )
    with np.load(basis_file) as pc:
        mean, basis, bounds = (pc[k] for k in ("mean", "basis", "bounds"))
    panels, images, ids = [], [], []
    metrics, details, totals = {}, [], {}
    for source in validation:
        views = camera_arrays(source)
        levels, pairs = brightness_pairs(views[0], cfg["evaluation"]["pairs_per_episode"])
        with np.load(source.path) as z:
            times = z["observation_ns"]
        acc = {"real_static_u2": np.zeros(3), "shift_and_minus1ev": np.zeros(3)}
        for j, (lo, hi) in enumerate(pairs):
            dim = np.stack([v[lo] for v in views])
            bright = np.stack([v[hi] for v in views])
            h, w = dim.shape[-2:]
            grid = (h // p - 1, w // p - 1)
            ref = tokens(bright[:, :-p, :-p])
            actual = tokens(dim[:1, p:, p:])
            # Same physical content, shifted by one patch and attenuated by one stop.
            attenuated = np.rint(
                np.clip((bright[:, p:, p:].astype("f4") - black) * 0.5 + black, 0, white)
            )
            aug = tokens(attenuated)
            acc["real_static_u2"] += match_patches(ref[:1], actual, grid, p, [WALL, MAT])
            acc["shift_and_minus1ev"] += match_patches(ref, aug, grid, p)
            if j == len(pairs) // 2:
                for kind, frame, raw in [("higher", hi, bright), ("lower", lo, dim)]:
                    full = tokens(raw).cpu().numpy()
                    projected = (full - mean) @ basis
                    colors = (projected - bounds[0]) / (bounds[1] - bounds[0])
                    panels.append(
                        np.rint(255 * np.clip(colors, 0, 1))
                        .astype("u1")
                        .reshape(2, h // p, w // p, 3)
                    )
                    images.append(np.stack([preview(a, black, white) for a in raw]))
                    ids.append(
                        dict(
                            episode=source.group_id,
                            brightness=kind,
                            frame=frame,
                            seconds=float((times[frame] - times[0]) * 1e-9),
                            wall_green_dn=float(levels[frame]),
                        )
                    )
        details.append(
            dict(
                episode=source.group_id,
                pairs=pairs,
                wall_low_high_dn=[[float(levels[i]), float(levels[j])] for i, j in pairs],
                scores={k: v.tolist() for k, v in acc.items()},
            )
        )
        for label in (source.suite, "all"):
            for name, value in acc.items():
                key = (label, name)
                totals[key] = totals.get(key, np.zeros(3)) + value
    for (label, name), (hits, count, error) in totals.items():
        metrics[f"visual/{label}/{name}_top1"] = float(hits / count)
        metrics[f"visual/{label}/{name}_mean_error_px"] = float(error / count)
    dest = folder / f"step-{step:06d}"
    np.savez_compressed(dest.with_suffix(".npz"), pca=np.stack(panels), preview=np.stack(images))
    dest.with_suffix(".json").write_text(
        json.dumps(dict(step=step, frames=ids, episodes=details, metrics=metrics), indent=2)
    )
    return metrics, []


def render(folder, step, baseline_step=0, dataset_label="fruit", initialization_label=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    def load(i):
        path = folder / f"step-{i:06d}"
        with np.load(path.with_suffix(".npz")) as f:
            arrays = {k: f[k] for k in f.files}
        return json.loads(path.with_suffix(".json").read_text()), arrays

    before, b = load(baseline_step)
    after, a = load(step)
    if before["frames"] != after["frames"]:
        raise ValueError("Before/after panels must use identical camera captures")
    first = {}
    for frame in after["frames"]:
        episode = frame["episode"]
        first.setdefault(episode.split("/")[0], episode)
    select = [i for i, frame in enumerate(after["frames"]) if frame["episode"] in first.values()]
    for camera, label in enumerate(("U2", "U3")):
        fig = plt.figure(figsize=(11, 3 * len(select)))
        grid = GridSpec(len(select), 3, figure=fig, wspace=0.035, hspace=0.14)
        for row, i in enumerate(select):
            meta = after["frames"][i]
            for col, img in enumerate(
                [a["preview"][i, camera], b["pca"][i, camera], a["pca"][i, camera]]
            ):
                ax = fig.add_subplot(grid[row, col])
                ax.imshow(img, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if row == 0:
                    ax.set_title(
                        [
                            "Recorded view",
                            initialization_label
                            or (
                                f"Before adaptation (step {baseline_step:,})"
                                if baseline_step
                                else "Simulation pretrained"
                            ),
                            f"After {step - baseline_step:,} additional updates"
                            if baseline_step
                            else f"After {step} real RAW updates",
                        ][col],
                        fontsize=12,
                        pad=8,
                    )
                if col == 0:
                    ax.set_ylabel(
                        f"{meta['episode'].split('/')[0]} · {meta['seconds']:.1f} s\n"
                        f"U2 wall {meta['wall_green_dn']:.0f} DN",
                        fontsize=9,
                    )
        fig.suptitle(f"{label} camera  |  Held-out {dataset_label} recordings", fontsize=17, y=0.98)
        fig.subplots_adjust(left=0.115, right=0.985, top=0.92, bottom=0.035)
        fig.savefig(folder / f"{label.lower()}-before-after.png", dpi=135, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render matched native RAW encoder panels")
    parser.add_argument("folder", type=Path)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--baseline-step", type=int, default=0)
    args = parser.parse_args()
    render(args.folder, args.step, args.baseline_step)
