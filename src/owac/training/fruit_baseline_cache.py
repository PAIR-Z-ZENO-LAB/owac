"""Prepare full DINOv3 patch grids and linear RGB inputs from the same recorded Bayer pairs."""

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from owac.raw.all_bench_distillation import load_inventory
from owac.raw.fruit_isp import DINO_SHA, IspDino, corp_rgb, linear_planes
from owac.training.teleop_policy_data import raw_views


@torch.inference_mode()
def main():
    import cv2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    inventory = Path(cfg["data"]["inventory"])
    if hashlib.sha256(inventory.read_bytes()).hexdigest() != cfg["data"]["inventory_sha256"]:
        raise ValueError("Fruit inventory changed")
    root = Path(cfg["root"]) / "cache"
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    folder = Path(cfg["assets"]["dino"])
    with (folder / "model.safetensors").open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != DINO_SHA:
            raise ValueError("Unexpected DINOv3 weights")
    net = IspDino(folder).requires_grad_(False).eval().cuda()
    size, linear_hw = cfg["model"]["image_size"], cfg["pi05"]["linear_shape"]
    sources = load_inventory(inventory)[0]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for src in sources[args.shard :: args.shards]:
            output = root / src.group_id
            output.mkdir(parents=True, exist_ok=True)
            sig = dict(
                episode=src.group_id,
                revision=cfg["data"]["dataset_revision"],
                dino_sha=DINO_SHA,
                image_size=size,
                linear_hw=linear_hw,
                isp="fixed-bggr-srgb8-v1",
            )
            receipt = output / "inputs.json"
            if (
                receipt.exists()
                and json.loads(receipt.read_text()) == sig
                and all((output / file).exists() for file in ("dino224.npy", "linear.npy"))
            ):
                continue
            left, right = raw_views(src)
            n = len(left)
            shp = (n, 2, (size // 16) ** 2, net.encoder.config.hidden_size)
            dino = np.lib.format.open_memmap(output / "dino224.partial.npy", "w+", np.float16, shp)
            linear = np.lib.format.open_memmap(
                output / "linear.partial.npy", "w+", np.float16, (n, 2, 3, *linear_hw)
            )
            for start in range(0, n, 16):
                stop = min(n, start + 16)
                images = np.stack((left[start:stop], right[start:stop]), 1).reshape(
                    -1, *left.shape[1:]
                )
                rgb = np.stack(list(pool.map(corp_rgb, images, [size] * len(images))))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, patches = net.encode_rgb(rgb)
                dino[start:stop] = patches.reshape(stop - start, *shp[1:]).half().cpu().numpy()
                values = np.stack([linear_planes(image, linear_hw) for image in images])
                linear[start:stop] = values.reshape(stop - start, 2, 3, *linear_hw)
            del dino, linear
            for key in ("dino224", "linear"):
                os.replace(output / (key + ".partial.npy"), output / (key + ".npy"))
            receipt.write_text(json.dumps(sig, indent=2))
            print(src.group_id, n, flush=True)


if __name__ == "__main__":
    main()
