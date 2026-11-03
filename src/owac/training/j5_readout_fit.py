"""Fit a regularized physical-action readout from RAW training trajectories only."""

import argparse
import copy
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from owac.evaluation.j5_regression_probe import compact_policy
from owac.raw.all_bench_distillation import load_inventory
from owac.training.j5_action_data import RawActionFrames, collate_actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    payload = torch.load(args.checkpoint, mmap=True, map_location="cpu", weights_only=False)
    cfg = copy.deepcopy(payload["config"])
    policy = compact_policy(args.checkpoint, "cuda:0")
    model = policy.model
    sources, _ = load_inventory(cfg["data"]["inventory"])
    sources = [
        s for s in sources if s.domain == "libero" and s.split == "train" and s.lighting == "Normal"
    ]
    dataset = RawActionFrames(
        sources, cfg, validation=True
    )  # evenly spaced anchors, still TRAIN sources
    loader = DataLoader(
        dataset,
        batch_size=16,
        num_workers=2,
        prefetch_factor=1,
        collate_fn=collate_actions,
        multiprocessing_context="spawn",
    )
    import swanlab

    swanlab.login(api_key=Path("/root/.config/owac/swanlab-api-key").read_text().strip())
    swanlab.merge_settings(swanlab.Settings(probe=swanlab.Settings.Probe(monitor=False)))
    swanlab.init(
        project="OWAC",
        workspace="Jeong",
        id="owac-j5-readout-fit-20261103",
        resume="allow",
        experiment_name="J5-动作读出｜解析校准｜LIBERO正常光训练集｜仅7维输出行｜冻结其余权重｜20261103",
        group="01 动作训练",
        job_type="readout-fit",
        config=cfg,
        logdir=str(args.output.parent / "swanlab-readout-fit"),
    )
    swanlab.define_metric("objective/*", section_name="01 训练目标")
    swanlab.define_metric("progress/*", section_name="04 运行资源")
    w = model.action_out_proj.weight.detach().double()[:7]
    b = model.action_out_proj.bias.detach().double()[:7]
    d = w.shape[1]
    hh = torch.zeros(d, d, device="cuda:0", dtype=torch.float64)
    hr = torch.zeros(d, 7, device="cuda:0", dtype=torch.float64)
    hs = torch.zeros(d, device="cuda:0", dtype=torch.float64)
    rs = torch.zeros(7, device="cuda:0", dtype=torch.float64)
    rr = torch.zeros(7, device="cuda:0", dtype=torch.float64)
    captured = []

    def capture(module, inputs):
        captured[:] = [inputs[0].detach()]

    hook = model.action_out_proj.register_forward_pre_hook(capture)
    count = 0
    anchors = 0
    torch.manual_seed(20261103)
    with torch.inference_mode():
        for i, batch in enumerate(loader):
            batch = {k: v.cuda(0) if torch.is_tensor(v) else v for k, v in batch.items()}
            noise = torch.randn_like(batch["actions"])
            time = model.sample_time(len(noise), noise.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(batch, noise=noise, time=time)
            valid = batch["action_valid"]
            h = captured[0][valid].double()
            target = (noise - batch["actions"])[valid, :7].double()
            residual = target - h @ w.T - b
            hh += h.T @ h
            hr += h.T @ residual
            hs += h.sum(0)
            rs += residual.sum(0)
            rr += residual.square().sum(0)
            count += len(h)
            anchors += len(noise)
            if i % 20 == 0:
                metrics = {
                    "progress/anchors": anchors,
                    "objective/readout_before_mse": float(rr.sum() / count / 7),
                }
                swanlab.log(metrics, step=anchors)
                print(metrics, flush=True)
        mean_h, mean_r = hs / count, rs / count
        covariance = hh / count - torch.outer(mean_h, mean_h)
        cross = hr / count - torch.outer(mean_h, mean_r)
        ridge = covariance.trace() / d
        delta = torch.linalg.solve(
            covariance + ridge * torch.eye(d, device="cuda:0", dtype=torch.float64), cross
        )
        intercept = mean_r - mean_h @ delta
        new_w = model.action_out_proj.weight.detach().float().cpu().clone()
        new_b = model.action_out_proj.bias.detach().float().cpu().clone()
        new_w[:7] += (delta.T).float().cpu()
        new_b[:7] += intercept.float().cpu()
        after = (
            rr / count
            - mean_r.square()
            - 2 * (delta * cross).sum(0)
            + (delta * (covariance @ delta)).sum(0)
        )
    hook.remove()
    state = dict(payload["ema"])
    state["action_out_proj.weight"] = new_w
    state["action_out_proj.bias"] = new_b
    cfg["adaptation"]["dense_modules"] = list(cfg["adaptation"]["dense_modules"]) + [
        "action_out_proj"
    ]
    receipt = dict(
        sources=len(sources),
        anchors=anchors,
        valid_action_rows=count,
        split="train",
        ridge=float(ridge),
        ridge_rule="mean eigenvalue of centered training hidden covariance",
        before_mse=float(rr.sum() / count / 7),
        after_mse=float(after.mean()),
        physical_weight_delta_norm=float(delta.norm()),
        bias_delta=intercept.cpu().tolist(),
        source_checkpoint=str(args.checkpoint),
        fit_rows=list(range(7)),
        gradient_updates=0,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            step=payload["step"],
            config=cfg,
            model=state,
            ema=state,
            inference_only=True,
            provenance=receipt,
        ),
        args.output,
    )
    args.output.with_suffix(".json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)
    swanlab.log(
        {
            "objective/readout_before_mse": receipt["before_mse"],
            "objective/readout_after_mse": receipt["after_mse"],
        },
        step=anchors,
    )
    swanlab.finish()


if __name__ == "__main__":
    main()
