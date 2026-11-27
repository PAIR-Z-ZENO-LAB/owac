"""Create and visualize training-only target masks from the fixed U2 view before pickup."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from owac.raw.all_bench_distillation import load_inventory
from owac.training.fruit_grasp import grasp_phase
from owac.training.teleop_policy_data import raw_views


def yellow(raw):
    # Each native BGGR cell contributes one red, two greens and one blue.
    b = raw[..., 0::2, 0::2].astype(np.float32)
    r = raw[..., 1::2, 1::2].astype(np.float32)
    return (r - b) / (np.maximum(r + b - 128, 1))


def labels(src, *, starting_goal=False):
    raw = raw_views(src)[0]
    x, y, w, h = (
        150,
        67,
        94,
        86,
    )  # U2 crop in Bayer cells, covering the starting mat above the round fruit.
    region = raw[:, 2 * y : 2 * (y + h), 2 * x : 2 * (x + w)]
    score = yellow(region).clip(-1, 1)
    scaled = ((score + 1) * 127.5).round().astype(np.uint8)
    threshold, initial = cv2.threshold(scaled[0], 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, components, area, _ = cv2.connectedComponentsWithStats(initial)
    if n < 2:
        raise ValueError(f"No initial target in {src.group_id}")
    first = components == 1 + np.argmax(area[1:, cv2.CC_STAT_AREA])
    visible = (scaled > threshold) & first[None]
    remaining, _ = grasp_phase(src, 1.5)
    visible[remaining < 0] = False
    weights = visible.sum((1, 2)).astype(np.float32) / first.sum()
    yy, xx = np.nonzero(first)
    cells = np.zeros((len(raw), raw.shape[1] // 16, raw.shape[2] // 16), np.float32)
    for row in range(len(raw)):
        iy, ix = np.nonzero(visible[row])
        if len(iy):
            np.add.at(cells[row], ((iy + y) // 8, (ix + x) // 8), 1 / len(iy))
    if starting_goal:
        # Supervise the demonstrated pickup site through the first close, including occlusion.
        weights = (remaining >= 0).astype(np.float32)
        cells = cells[:1] * weights[:, None, None]
    centre = [float((xx + x + 0.5).mean() * 2), float((yy + y + 0.5).mean() * 2)]
    preview = raw[0, 0::2, 0::2].astype(float)
    preview = (
        255
        * np.sqrt(np.maximum(preview - 64, 0) / max(np.quantile(preview, 0.99) - 64, 1)).clip(0, 1)
    ).astype(np.uint8)
    preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2RGB)
    mask = np.zeros(preview.shape[:2], np.uint8)
    mask[y : y + h, x : x + w] = first
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(preview, contours, -1, (255, 90, 35), 1)
    return cells, weights, centre, Image.fromarray(preview)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--starting-goal", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources, _ = load_inventory(args.inventory)
    manifest = {}
    for suite in ("fruit01", "fruit02"):
        page = Image.new("RGB", (5 * 260, 2 * 202), "white")
        draw = ImageDraw.Draw(page)
        for i, s in enumerate(
            sorted((s for s in sources if s.suite == suite), key=lambda s: s.group_id)
        ):
            cells, w, centre, im = labels(s, starting_goal=args.starting_goal)
            dest = args.output / (s.group_id + ".npz")
            dest.parent.mkdir(exist_ok=True)
            np.savez_compressed(dest, mass=cells.astype(np.float16), weight=w)
            x = i % 5 * 260
            y = i // 5 * 202
            draw.text((x + 2, y + 1), s.group_id + " " + s.split, fill="black")
            page.paste(im.resize((256, 186)), (x, y + 16))
            manifest[s.group_id] = dict(
                split=s.split, initial_xy=centre, supervised_frames=int((w > 0).sum())
            )
        page.save(args.output / (suite + ".jpg"), quality=90)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
