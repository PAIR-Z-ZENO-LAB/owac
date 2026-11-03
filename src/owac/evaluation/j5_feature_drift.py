"""Measure RAW-token and bridge-output changes on fixed replay observations."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from owac.training.j5_lora import ResidualProjector
from owac.training.raw_ssl import RawEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    torch.set_num_threads(4)
    cfg = json.loads((root / "production.json").read_text())
    j5 = torch.load(cfg["assets"]["j5"], mmap=True, weights_only=False, map_location="cpu")
    pairs = []
    for filename in ["initial-003000.pt", "step-004000.pt"]:
        state = torch.load(
            root / "normal" / filename, mmap=True, weights_only=False, map_location="cpu"
        )["ema"]
        encoder = RawEncoder(j5["config"]["model"])
        encoder.load_state_dict(
            {
                n.removeprefix("raw_encoder."): v
                for n, v in state.items()
                if n.startswith("raw_encoder.")
            }
        )
        bridge = ResidualProjector(nn.Linear(j5["config"]["model"]["width"], 2048))
        bridge.load_state_dict(
            {
                n.removeprefix("raw_projector."): v
                for n, v in state.items()
                if n.startswith("raw_projector.")
            }
        )
        pairs.append((encoder.cuda(1).eval(), bridge.cuda(1).eval()))
    paths = json.loads((root / "weight-audit/probe-index.json").read_text())["paths"]
    rows = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for path in paths:
            with np.load(path) as z:
                x = torch.from_numpy(z["raw"]).cuda(1).permute(0, 3, 1, 2).float() / 255
            a = pairs[0][0](x)[1]
            b = pairs[1][0](x)[1]
            projected = pairs[0][1](a)
            versions = {"vision": pairs[0][1](b), "bridge": pairs[1][1](a), "both": pairs[1][1](b)}

            def metrics(a, b):
                a, b = a.float(), b.float()
                return dict(
                    cosine=float(F.cosine_similarity(a, b, dim=-1).mean()),
                    relative_rms=float((b - a).norm() / a.norm()),
                    spatial_variance_ratio=float(b.var(dim=1).sum() / a.var(dim=1).sum()),
                )

            rows.append(
                dict(
                    path=path,
                    raw=metrics(a, b),
                    **{n: metrics(projected, v) for n, v in versions.items()},
                )
            )
    (root / "weight-audit/feature-drift.json").write_text(json.dumps(rows, indent=2))
    print(
        json.dumps(
            {
                g: {k: float(np.mean([r[g][k] for r in rows])) for k in rows[0][g]}
                for g in ["raw", "vision", "bridge", "both"]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
