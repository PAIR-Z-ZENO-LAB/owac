"""Export the J5 global and patch descriptors for one public LIBERO or RoboTwin frame."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from owac.raw.all_bench_distillation import DOMAIN_RAW_KEYS
from owac.training.raw_ssl import build_encoder
from owac.training.raw_ssl_data import RawViews


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--domain", choices=("libero", "robotwin2"), default="libero")
    parser.add_argument("--camera", help="NPZ member, default is the first camera in the domain")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    if args.frame < 0:
        parser.error("Frame index must be nonnegative")
    keys = DOMAIN_RAW_KEYS[args.domain]
    camera = args.camera or keys[0]
    if camera not in keys:
        parser.error(f"Expected a {args.domain} camera from {keys}")
    ck = torch.load(args.encoder, map_location="cpu", weights_only=True, mmap=True)
    model = build_encoder(ck["config"]["model"]).to(args.device).eval()
    model.load_state_dict(ck["encoder"], strict=True)
    reader = RawViews([], ck["config"])
    try:
        frame = reader.frame(
            SimpleNamespace(path=args.episode, domain=args.domain), args.frame, camera
        )
    finally:
        reader.close()
    batch = F.interpolate(frame[None], size=(256, 256), mode="bilinear", align_corners=False)
    features, patches = model(batch.to(args.device))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, cls=features[0].cpu().numpy(), patches=patches[0].cpu().numpy())
    print(
        json.dumps(
            {
                "step": ck["step"],
                "cls": list(features.shape[1:]),
                "patches": list(patches.shape[1:]),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
