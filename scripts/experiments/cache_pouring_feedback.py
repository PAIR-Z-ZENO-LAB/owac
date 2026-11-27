"""Compute state-independent paired-RAW features and original pouring predictions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from owac.frameworks.raw_compact.pouring_feedback import stereo_grid
from owac.raw.all_bench_distillation import load_inventory
from owac.runtime.teleop_history import causal_history_indices
from owac.training.compact_data import normalize_state
from owac.training.teleop_policy_data import raw_views
from owac.training.teleop_recovery import fixed_language, load_export, predict_cached


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("parent", type=Path)
    parser.add_argument("--batch", type=int, default=12)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, step = load_export(args.parent)
    cfg = model.cfg
    language = fixed_language(model)
    dest = args.root / "feedback-cache"
    dest.mkdir(exist_ok=True)
    receipt = dest / "source.json"
    with args.parent.open("rb") as stream:
        signature = dict(step=step, sha256=hashlib.file_digest(stream, "sha256").hexdigest())
    if receipt.exists() and json.loads(receipt.read_text()) != signature:
        raise ValueError("Parent differs from cached policy")
    receipt.write_text(json.dumps(signature, indent=2))
    for episode in load_inventory(args.root / "aligned/inventory.json")[0]:
        path = dest / (episode.group_id + ".npz")
        if path.is_file():
            continue
        path.parent.mkdir(exist_ok=True)
        views = raw_views(episode)
        with np.load(episode.path) as data:
            q = torch.from_numpy(normalize_state(data["state"], cfg["statistics"])).cuda()
            ix = causal_history_indices(data["observation_ns"], [0, 3, 6], 1e9 / 30)
        bank = {"base": [], "grid": [], "visual": []}
        for start in range(0, len(q), args.batch):
            s = slice(start, start + args.batch)
            raw = torch.from_numpy(np.stack([v[s] for v in views], 1).astype("f4")).cuda()
            size = len(raw)
            batch = dict(
                raw=raw,
                state=q[s],
                token_mask=language[2].expand(size, -1),
                tokens=torch.tensor(cfg["task_token_ids"], device="cuda").expand(size, -1),
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                rep, visual, patches = model.encode(batch, return_patches=True)
                direct = model.action_generator(model.action_generator.prepare_prefix(rep))
                base = [
                    direct,
                    *(predict_cached(model, visual, q[ix[s, k]], language) for k in (1, 2)),
                ]
                grid = stereo_grid(patches, cfg, size)
            for key, value in (("grid", grid), ("visual", visual), ("base", torch.stack(base, 1))):
                bank[key].append(value.to(torch.bfloat16).view(torch.int16).cpu().numpy())
        tmp = path.with_suffix(".partial.npz")
        np.savez(tmp, **{key: np.concatenate(value) for key, value in bank.items()})
        tmp.replace(path)
        print(episode.group_id, episode.num_frames, flush=True)


if __name__ == "__main__":
    main()
