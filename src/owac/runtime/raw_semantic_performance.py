"""Zero-update measured microbatch comparison on the next real sampler windows."""

import argparse
import random
import time
from pathlib import Path

import torch

from owac.training.raw_jepa_train import epoch_batches, load_inventory, write_json
from owac.training.raw_semantic import build_semantic, semantic_loss
from owac.training.raw_semantic_train import DinoTargets, PairedWindows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--large-only", action="store_true")
    args = p.parse_args()
    state = torch.load(args.run_root / "latest.pt", map_location="cpu", weights_only=False)
    c = state["config"]
    torch.set_num_threads(4)
    m = build_semantic(c).cuda()
    m.load_state_dict(state["encoder"])
    names = ("log_conv", "content_gate", "semantic_projection", "depth_logits")
    groups = [
        {"params": [p for n, p in m.named_parameters() if any(k in n for k in names)]},
        {"params": [p for n, p in m.named_parameters() if not any(k in n for k in names)]},
    ]
    optimizer = torch.optim.AdamW(groups)
    optimizer.load_state_dict(state["optimizer"])
    source_step, epoch, cursor = state["step"], state["epoch"], state["cursor"]
    del state
    teacher = DinoTargets(c["semantic"]["teacher"], "cuda")
    sources, _ = load_inventory(c["data"]["inventory"])
    ds = PairedWindows(sources, c)
    batches = epoch_batches(ds.keys("train"), 32, c["training"]["seed"], epoch)[cursor : cursor + 8]
    rows = []
    cases = [(8, 16)] if args.large_only else [(4, 2), (4, 8), (8, 8)]
    for threads, micro in cases:
        torch.set_num_threads(threads)
        for i, keys in enumerate(batches):
            random.seed(9000 + i)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.monotonic()
            mine = keys[::2]
            total = torch.zeros((), device="cuda")
            load_seconds = 0
            for off in range(0, len(mine), micro):
                before = time.monotonic()
                raw, rgb = ds.paired(mine[off : off + micro], "cuda", augment=True)
                load_seconds += time.monotonic() - before
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    target = teacher(rgb)
                    f = m(raw)
                    loss, _ = semantic_loss(f, target, **c["loss"])
                    (loss * micro / len(mine)).backward()
                total += loss.detach() * micro / len(mine)
            torch.cuda.synchronize()
            rows.append(
                dict(
                    threads=threads,
                    micro=micro,
                    batch=i,
                    seconds=time.monotonic() - start,
                    load_seconds=load_seconds,
                    loss=float(total),
                    peak=torch.cuda.max_memory_allocated(),
                )
            )
            print(rows[-1], flush=True)
    write_json(
        args.output,
        dict(
            source_step=source_step,
            new_optimizer_updates=0,
            rows=rows,
            note=(
                "one GPU; actual Adam state resident; same next eight sampler batches/RNG; "
                "no optimizer step; first batch warmup per setting"
            ),
        ),
    )


if __name__ == "__main__":
    main()
