"""Fit an absolute visual action anchor against measured and self-induced pouring drift."""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import swanlab
import torch
from replay_pouring_feedback import PouringReplay
from torch.nn import functional as F
from train_pouring_feedback import write

from owac.frameworks.raw_compact.factory import make_policy
from owac.raw.all_bench_distillation import load_inventory
from owac.training.compact_train import atomic_save
from owac.training.fruit_grasp import NeroHand
from owac.training.pouring_feedback_data import PouringBank, compare_actions, score_actions
from owac.training.teleop_recovery import fixed_language, load_export, predict_cached


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for k in ("root", "original", "urdf"):
        parser.add_argument(k, type=Path)
    args = parser.parse_args()
    root = args.root
    out = root / "anchor"
    out.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(20261125)
    head, step = load_export(root / "run/policy-final.pt")
    cfg = head.cfg.copy()
    cfg["model"] = dict(cfg["model"], pouring_absolute_anchor=True)
    model = make_policy(cfg)
    missing, unexpected = model.load_state_dict(head.state_dict(), strict=False)
    if unexpected or any(
        not k.startswith(("anchor_", "trust_logit", "disagreement")) for k in missing
    ):
        raise ValueError("Unexpected feedback parent")
    model.eval().requires_grad_(False).cuda()
    for m in (model.anchor_head, model.anchor_grid):
        m.requires_grad_(True).train()
    params = [
        *model.anchor_head.parameters(),
        *model.anchor_grid.parameters(),
        model.trust_logit.requires_grad_(),
        model.disagreement.requires_grad_(),
    ]
    episodes, _ = load_inventory(root / "aligned/inventory.json")
    train = [e for e in episodes if e.split == "train"]
    dev = [e for e in episodes if e.split == "validation"]
    hand = NeroHand(args.urdf, "cpu")
    bank = PouringBank(train, root / "feedback-cache", cfg, hand)
    rb = PouringReplay(train, root, cfg, json.loads((root / "servo-response.json").read_text()))
    original, _ = load_export(args.original)
    language = fixed_language(head)
    updates = 3200
    spec = dict(
        updates=updates,
        seed=20261125,
        batch=96,
        train=bank.names,
        development=[e.group_id for e in dev],
        parent_step=step,
        device=torch.cuda.get_device_name(),
        execution_horizon=4,
        target="Absolute expert command from two RAW views, blended with measured feedback",
        perturbations=(
            "Train episode commands recursively fed to train-fitted servo, gap 1, prefix 4"
        ),
        selection=(
            "Fixed final update; earlier development diagnostics motivated this second experiment"
        ),
    )
    cfg["pouring_anchor"] = spec
    write(out / "protocol.json", spec)
    run = swanlab.init(
        workspace=os.environ.get("SWANLAB_WORKSPACE"),
        mode=os.environ.get("SWANLAB_MODE", "local"),
        project=os.environ.get("SWANLAB_PROJECT", "OWAC"),
        public=False,
        group="real-pouring-20261123",
        job_type="action-policy",
        id="pouring-visual-anchor-20261124",
        name="动作补训｜OWAC｜Pouring｜视觉绝对目标与偏差恢复｜3200步｜20261124",
        config=spec,
        resume="allow",
        logdir=str(out / "swanlog"),
    )
    write(out / "swanlab.json", dict(url=run.url))
    with torch.no_grad():
        drift = [
            torch.zeros_like(rb.q),
            rb.rollout(original, language, 4),
            rb.rollout(head, language, 4),
        ]
        drift = torch.stack(drift)
    del original
    optimizer = torch.optim.AdamW(params, lr=0.0004)
    start = 0
    if (out / "latest.pt").is_file():
        saved = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.cuda.set_rng_state(saved["cuda_rng"])
        start = saved["update"]
    running = []
    for i in range(start + 1, updates + 1):
        rng = np.random.default_rng(20261125 + i)
        ix = bank.indices(rng, 96)
        hist = rb.history_index[ix]
        mode = torch.as_tensor(rng.choice([0, 0, 1, 2], len(ix)), device="cuda")
        fraction = torch.tensor(rng.random((len(ix), 1)), device="cuda", dtype=torch.float32)
        q = (rb.q[ix] + fraction * drift[mode, ix] - rb.mean) / rb.sd
        past = (rb.q[hist] + fraction[:, None] * drift[mode[:, None], hist] - rb.mean) / rb.sd
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                base = predict_cached(head, rb.visual[ix], q, language)
                previous = head.feedback(base, bank.grid[ix], q, past)
            absolute = model.absolute(bank.grid[ix])
            norm = model.blend(previous, absolute, q)
        ok = bank.valid[ix]
        gt = bank.target[ix]
        diff = (bank.physical(norm, q) - gt) / model.action_half
        mixed = F.smooth_l1_loss(diff[ok], torch.zeros_like(diff[ok]), beta=0.1, reduction="none")
        direct = (absolute - gt) / model.q_scale
        aux = F.smooth_l1_loss(direct[ok], torch.zeros_like(direct[ok]), beta=0.1)
        loss = (mixed[:, :7].mean() + mixed[:, 7].mean() + aux) / 3
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        optimizer.param_groups[0]["lr"] = (
            0.0004 * min(1, i / 80) * (0.08 + 0.92 * (1 + math.cos(math.pi * i / updates)) / 2)
        )
        optimizer.step()
        running.append(float(loss.detach()))
        if i % 25 == 0 or i == 1:
            value = float(np.mean(running))
            swanlab.log({"train/loss": value}, step=i)
            with (out / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(dict(step=i, loss=value)) + "\n")
            running.clear()
        if i % 400 == 0:
            atomic_save(
                dict(
                    config=cfg,
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    cuda_rng=torch.cuda.get_rng_state(),
                    step=step + i,
                    update=i,
                ),
                out / "latest.pt",
            )
            print("updates", i, flush=True)
    model.eval()
    atomic_save(
        dict(config=cfg, model=model.state_dict(), step=step + updates), out / "policy-final.pt"
    )
    del rb, bank, drift
    validation = PouringBank(dev, root / "feedback-cache", cfg, hand)
    results = {}
    metrics = {}
    for delay, frames in enumerate((0, 3, 6)):
        _, actions = compare_actions(validation, model, delay)
        results[str(frames)] = score_actions(validation, actions, hand)
        metrics.update(
            {f"validation/delay{frames}/{k}": v for k, v in results[str(frames)]["summary"].items()}
        )
    write(out / "validation.json", results)
    swanlab.log(metrics, step=updates + 1)
    swanlab.finish()


if __name__ == "__main__":
    main()
