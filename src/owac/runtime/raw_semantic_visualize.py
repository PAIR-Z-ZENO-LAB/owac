"""Fixed held-source query affinity evidence: RAW encoder versus RGB DINOv3."""

import argparse
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from owac.training.raw_jepa_train import load_inventory, write_json
from owac.training.raw_semantic import build_semantic
from owac.training.raw_semantic_train import DinoTargets, PairedWindows


def export_queries(encoder, output, device="cpu"):
    args = SimpleNamespace(encoder=Path(encoder), output=Path(output), device=device)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    state = torch.load(args.encoder, map_location="cpu", weights_only=False)
    config = state["config"]
    model = build_semantic(config).eval()
    model.load_state_dict(state["encoder"])
    model.to(args.device)
    step = state["step"]
    del state
    teacher = DinoTargets(config["semantic"]["teacher"], args.device)
    sources, _ = load_inventory(config["data"]["inventory"])
    config["training"]["working_cache_sources"] = 2
    ds = PairedWindows(sources, config)
    selected = [
        sorted(
            [s for s in ds.sources.values() if s.split == "validation" and s.lighting == light],
            key=lambda s: s.group_id,
        )[0]
        for light in ("Normal", "Low", "ExtremeLow", "Over", "ExtremeOver")
    ]
    arrays = {}
    manifest = []
    with torch.inference_mode():
        for i, s in enumerate(selected):
            end = (s.num_frames - 1 + 5) // 2
            raw, rgb = ds.paired([(s.group_id, end)], args.device, augment=False)
            student = model(raw[:1]).float()
            target = teacher(rgb[:1]).float()
            for name, features in [("raw", student), ("dino", target)]:
                features = F.normalize(features[0], dim=-1)
                arrays[f"{i}_{name}_affinity"] = (features @ features.T).cpu().numpy()
            arrays[f"{i}_rgb"] = rgb[0].permute(1, 2, 0).cpu().numpy()
            arrays[f"{i}_raw_preview"] = raw[0, -1].permute(1, 2, 0).cpu().numpy() ** (1 / 2.2)
            manifest.append(dict(source=s.group_id, lighting=s.lighting, endpoint=end, camera=0))
            print(i, s.lighting, flush=True)
    np.savez_compressed(args.output / "affinities.npz", **arrays)
    write_json(
        args.output / "manifest.json",
        dict(
            step=step,
            samples=manifest,
            queries=[[4, 4], [8, 8], [12, 12]],
            protocol=(
                "fixed validation source per lighting, middle endpoint, camera0; full FOV; "
                "RAW6 frames resized224, DINO RGB endpoint256; RAW display gamma only; "
                "affinity cosine, no per-image rescaling; fixed query positions "
                "are not semantic labels"
            ),
        ),
    )
    renderer = (
        Path(__file__).resolve().parents[3] / "scripts/analysis/render_raw_semantic_queries.py"
    )
    subprocess.run([sys.executable, str(renderer), "--root", str(args.output)], check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    export_queries(args.encoder, args.output, args.device)


if __name__ == "__main__":
    main()
