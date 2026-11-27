"""Train a pouring-specific feedback readout on the original episode split, then freeze it."""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import swanlab
import torch
from torch.nn import functional as F

from owac.frameworks.raw_compact.factory import make_policy
from owac.raw.all_bench_distillation import load_inventory
from owac.training.compact_train import atomic_save
from owac.training.fruit_grasp import NeroHand
from owac.training.pouring_feedback_data import PouringBank, compare_actions, score_actions


def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("parent", type=Path)
    p.add_argument("urdf", type=Path)
    args = p.parse_args()
    root = args.root
    out = root / "run"
    out.mkdir(exist_ok=True)
    cache = root / "feedback-cache"
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    seed, updates, batch_size = 20261124, 3000, 96
    torch.manual_seed(seed)
    ck = torch.load(args.parent, map_location="cpu", mmap=True, weights_only=False)
    cfg = ck["config"]
    cfg["model"].update(kind="pouring_feedback_regression", state_history_lags=[1, 3, 6, 12, 24])
    model = make_policy(cfg)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    if unexpected or any(not k.startswith(("grid_", "feedback_head.")) for k in missing):
        raise ValueError("Unexpected parent architecture")
    model = model.eval().requires_grad_(False).cuda()
    head = [model.feedback_head, model.grid_readout]
    for module in head:
        module.requires_grad_(True).train()
    source, _ = load_inventory(root / "aligned/inventory.json")
    train_src = [s for s in source if s.split == "train"]
    dev_src = [s for s in source if s.split == "validation"]
    hand = NeroHand(args.urdf, "cpu")
    train = PouringBank(train_src, cache, cfg, hand)
    for center, scale, bank in zip(
        model.grid_center.split(1),
        model.grid_scale.split(1),
        train.grid.unbind(1),
        strict=True,
    ):
        deviation, mean = torch.std_mean(bank.float(), dim=0)
        center.copy_(mean)
        scale.copy_(deviation.clamp_min(torch.finfo(torch.float32).eps))
    spec = dict(
        parent=json.loads((cache / "source.json").read_text()),
        updates=updates,
        seed=seed,
        batch_size=batch_size,
        training=[s.group_id for s in train_src],
        validation=[s.group_id for s in dev_src],
        dataset_revision=cfg["data"]["dataset_revision"],
        input="Current twin BG10, measured 7-axis joints and aperture, past 1/3/6/12/24 frames",
        label="Timestamp-aligned expert mux command in rad/m, 12 future rows at 30 Hz",
        sampling="1/2 uniform frames, 1/4 aperture transitions, 1/4 tilted carry; uniform episodes",
        delay_frames_fit=[0, 3],
        delay_frames_evaluation=[0, 3, 6],
        execution_horizon=4,
        selection="Fixed final 3000 updates; six development episodes first scored after export",
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        device=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
    )
    cfg["pouring_feedback"] = spec
    write(out / "protocol.json", spec)
    sw = swanlab.init(
        workspace=os.environ.get("SWANLAB_WORKSPACE"),
        mode=os.environ.get("SWANLAB_MODE", "local"),
        project=os.environ.get("SWANLAB_PROJECT", "OWAC"),
        public=False,
        group="real-pouring-20261123",
        job_type="action-policy",
        id="pouring-feedback-20261124",
        name="动作补训｜OWAC｜Pouring｜握杯与放回反馈｜3k｜20261124",
        config=spec,
        resume="allow",
        logdir=str(out / "swanlog"),
    )
    write(out / "swanlab.json", dict(url=sw.url))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=3e-4)
    start = 0
    if (out / "latest.pt").is_file():
        saved = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        start = saved["update"]
        torch.cuda.set_rng_state(saved["cuda_rng"])
    names = ("arm", "jaw")
    losses = []
    for i in range(start + 1, updates + 1):
        rng = np.random.default_rng(seed + i)
        ix = train.indices(rng, batch_size)
        delay = torch.as_tensor(rng.choice([0, 0, 1], size=batch_size), device="cuda")
        q = train.q[ix, delay]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model.correct(train.base[ix, delay], train.grid[ix], q, train.past[ix, delay])
        error = (train.physical(pred, q) - train.target[ix]) / (train.scale / 2)
        ok = train.valid[ix]
        # Balance seven joint dimensions against the one aperture channel in physical units.
        errors = F.smooth_l1_loss(error, torch.zeros_like(error), beta=0.1, reduction="none")[ok]
        arm, jaw = errors[:, :7].mean(), errors[:, 7].mean()
        loss = (arm + jaw) / 2
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        lr = 3e-4 * min(i / 80, 1) * (0.05 + 0.95 * (1 + math.cos(math.pi * i / updates)) / 2)
        optimizer.param_groups[0]["lr"] = lr
        optimizer.step()
        losses.append((float(arm.detach()), float(jaw.detach())))
        if i % 25 == 0 or i == 1:
            metrics = {
                "train/" + k: v for k, v in zip(names, np.mean(losses, 0).tolist(), strict=True)
            }
            metrics["train/lr"] = lr
            swanlab.log(metrics, step=i)
            with (out / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(step=i, **metrics)) + "\n")
            losses.clear()
        if i % 500 == 0:
            atomic_save(
                dict(
                    config=cfg,
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    cuda_rng=torch.cuda.get_rng_state(),
                    parent_step=ck["step"],
                    update=i,
                    step=ck["step"] + i,
                ),
                out / "latest.pt",
            )
            print("updates", i, flush=True)
    model.eval()
    if not (out / "policy-final.pt").exists():
        atomic_save(
            dict(config=cfg, model=model.state_dict(), step=ck["step"] + updates),
            out / "policy-final.pt",
        )
    del train
    dev = PouringBank(dev_src, cache, cfg, hand)
    results, metrics = {}, {}
    for delay, frames in enumerate((0, 3, 6)):
        base, corrected = compare_actions(dev, model, delay)
        for label, actions in (("original", base), ("candidate", corrected)):
            report = score_actions(dev, actions, hand)
            results[f"{label}_{frames}"] = report
            metrics.update(
                {f"validation/{label}_delay{frames}/{k}": v for k, v in report["summary"].items()}
            )
    write(out / "validation.json", results)
    swanlab.log(metrics, step=updates + 1)
    swanlab.finish()
    print(json.dumps({k: v["summary"] for k, v in results.items()}), flush=True)


if __name__ == "__main__":
    main()
