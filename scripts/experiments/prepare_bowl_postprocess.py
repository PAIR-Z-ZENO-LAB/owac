"""Extract scene, joint and action references for the bowl release and smoothing fits."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from owac.raw.all_bench_distillation import load_inventory
from owac.training.fruit_grasp import NeroHand
from owac.training.teleop_recovery import fixed_language, load_export
from owac.training.teleop_recovery_replay import RecoveryBank


@torch.inference_mode()
def main(parent, dest, urdf):
    torch.set_num_threads(2)
    feats = dest / "features"
    feats.mkdir(parents=True, exist_ok=True)
    model, _ = load_export(parent / "motion-history/policy-best.pt")
    text = fixed_language(model)
    hand = NeroHand(urdf, "cpu")
    geometry = {k: getattr(hand, k).tolist() for k in ("xyz", "rot", "tip")}
    (dest / "geometry.json").write_text(json.dumps(geometry))
    tracking = json.loads((parent / "field-tracking.json").read_text())
    episodes, _ = load_inventory(parent / "aligned/inventory.json")
    for split in ("train", "validation"):
        sources = [e for e in episodes if e.split == split]
        bank = RecoveryBank(sources, parent / "visual", model.cfg, tracking)
        bank.load_spatial(parent / "spatial")
        bank.configure_output(model.cfg)
        bank.history_rows(model.cfg["model"]["state_history_lags"])
        chunks = []
        for ix in torch.arange(len(bank.q), device="cuda").split(120):
            chunks.append(
                bank.predict(model, text, ix, bank.q[ix], bank.q[bank.history_index[ix]])
                .cpu()
                .numpy()
            )
        plans = np.concatenate(chunks)
        for e, start, size in zip(sources, bank.starts, bank.sizes, strict=True):
            key = e.group_id.replace("/", "-")
            with np.load(e.path) as data:
                q, cmd = data["state"], data["action"]
                t = (data["observation_ns"] - data["observation_ns"][0]) / 1e9
            scenes = []
            for ix in torch.arange(start, start + size, device="cuda").split(64):
                x = bank.maps[ix].cpu().float().flatten(0, 1)
                scenes.append(F.adaptive_avg_pool2d(x, (3, 4)).reshape(len(ix), -1).cpu().numpy())
            np.savez_compressed(
                feats / (key + ".npz"),
                t=t,
                q=q,
                command=cmd,
                xyz=hand.pose(torch.from_numpy(q[:, :7]))[0].numpy(),
                scene=np.concatenate(scenes).astype("f2"),
            )
            np.save(feats / (key + "-plans.npy"), plans[start : start + size])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("parent", "experiment", "urdf"):
        p.add_argument(k, type=Path)
    a = p.parse_args()
    main(a.parent, a.experiment, a.urdf)
