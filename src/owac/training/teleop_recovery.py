"""Fixed visual observations for state-feedback training and replay on the real robot data."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from owac.frameworks.raw_compact.factory import make_policy
from owac.raw.all_bench_distillation import load_inventory
from owac.training.compact_data import normalize_state
from owac.training.teleop_policy_data import raw_views


def load_export(path):
    payload = torch.load(path, weights_only=False, mmap=True, map_location="cpu")
    model = make_policy(payload["config"])
    model.load_state_dict(payload["model"], strict=True)
    return model.eval().cuda(), payload["step"]


def fixed_language(model):
    cfg = model.cfg
    tokens = torch.tensor([cfg["task_token_ids"]], device="cuda")
    mask = torch.tensor([cfg["task_token_mask"]], dtype=torch.bool, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        language, pooled, _ = model.conditioning(
            dict(tokens=tokens, token_mask=mask, state=torch.zeros(1, 8, device="cuda"))
        )
    return language, pooled, mask


def predict_cached(model, visual, state_norm, text):
    language, pooled, mask = text
    size = len(state_norm)
    rep = model.condition_visual(
        {"token_mask": mask.expand(size, -1)},
        visual,
        language.expand(size, -1, -1),
        pooled.expand(size, -1),
        model.state_encoder(state_norm),
    )
    return model.action_generator(model.action_generator.prepare_prefix(rep))


@torch.inference_mode()
def cache(args):
    torch.set_num_threads(2)
    model, step = load_export(args.parent)
    cfg = model.cfg
    text = fixed_language(model)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "source.json"
    with args.parent.open("rb") as source:
        signature = dict(step=step, sha256=hashlib.file_digest(source, "sha256").hexdigest())
    if manifest.exists() and json.loads(manifest.read_text()) != signature:
        raise ValueError("Cached RAW features belong to a different policy")
    manifest.write_text(json.dumps(signature, indent=2))
    for episode in load_inventory(args.inventory)[0]:
        dest = args.output / (episode.group_id + ".npy")
        if dest.is_file():
            continue
        dest.parent.mkdir(exist_ok=True)
        views = raw_views(episode)
        with np.load(episode.path) as aligned:
            q = torch.from_numpy(normalize_state(aligned["state"], cfg["statistics"])).cuda()
        temporary = dest.with_suffix(".partial.npy")
        for first in range(0, len(q), args.batch):
            sl = slice(first, first + args.batch)
            raw = torch.from_numpy(np.stack([v[sl] for v in views], axis=1).astype("f4")).cuda()
            bs = len(raw)
            batch = dict(
                raw=raw,
                state=q[sl],
                token_mask=text[2].expand(bs, -1),
                tokens=torch.tensor(cfg["task_token_ids"], device="cuda").expand(bs, -1),
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prefix, visual = model.encode(batch)
                if first == 0:
                    expected = model.action_generator(model.action_generator.prepare_prefix(prefix))
                    actual = predict_cached(model, visual, q[sl], text)
                    # Only batched T5 rounding differs when caching one copy of the instruction.
                    diff = float((expected - actual).abs().max())
                    if not np.isfinite(diff):
                        raise ValueError("Nonfinite production-to-cache replay")
                    bank = np.lib.format.open_memmap(
                        temporary, mode="w+", dtype=np.int16, shape=(len(q), *visual.shape[1:])
                    )
            bank[sl] = visual.to(torch.bfloat16).view(torch.int16).cpu().numpy()
        bank.flush()
        del bank
        temporary.replace(dest)
        print(episode.group_id, len(q), "normalized parity max", diff, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch", type=int, default=12)
    cache(parser.parse_args())
