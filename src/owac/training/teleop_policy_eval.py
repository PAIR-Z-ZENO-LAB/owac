"""Physical-unit action evaluation and offline inference for the fruit policy."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from owac.frameworks.raw_compact.factory import make_policy
from owac.raw.all_bench_distillation import load_inventory
from owac.training.compact_actions import decode_actions
from owac.training.compact_data import normalize_state, unnormalize_actions
from owac.training.teleop_policy_data import gripper_travel, raw_views


@torch.inference_mode()
def evaluate(
    model,
    bank,
    *,
    method="policy",
    linear=None,
    rank=0,
    world=1,
    batch_size=8,
    predictions=None,
    exposure=0,
):
    horizon = bank.cfg["model"]["action_horizon"]
    sums = torch.zeros((len(bank.sources), 13), dtype=torch.float64, device=bank.device)
    episodes = bank.tensor(bank.episode)
    anchors = bank.cfg.get("evaluation", {}).get("selection_anchors_per_episode")
    if anchors:
        selection = torch.zeros(len(episodes), dtype=torch.bool, device=bank.device)
        for start, size in zip(bank.starts, bank.sizes, strict=True):
            offsets = np.linspace(0, size - 1, min(anchors, size), dtype=np.int64)
            selection[bank.tensor(start + offsets)] = True
        selected_sums = torch.zeros_like(sums[:, :2]).contiguous()
    for ix in torch.arange(rank, len(bank.episode), world, device=bank.device).split(batch_size):
        if method == "hold":
            physical = bank.state[ix, None, :].expand(-1, horizon, -1)
            relative = physical - (bank.state[ix] * bank.relative_mask)[:, None, :]
            pred = 2 * (relative - bank.low) / bank.scale - 1
        elif method == "linear":
            state = bank.state_norm[ix]
            x = torch.cat((state, torch.ones_like(state[:, :1])), 1)
            pred = torch.einsum("bi,hij->bhj", x, linear)
            physical = bank.physical(pred, ix)
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model.sample(
                    bank.batch(ix, wrong=method == "wrong_views", exposure=exposure)
                )
            physical = bank.physical(pred, ix)
        if predictions is not None:
            predictions[ix.cpu().numpy()] = physical.cpu().numpy()
        valid = bank.valid[ix]
        sq = (pred - bank.normalized[ix]).square().mean(-1)
        error = (physical - bank.absolute[ix]).abs()
        degrees = torch.rad2deg(error[:, :, :7].mean(-1))
        width = error[:, :, 7] * 1000
        issued_width = (physical[..., 7].clamp_min(0) - bank.absolute[ix, :, 7]).abs() * 1000
        movement = gripper_travel(bank.absolute[ix], valid)
        values = torch.stack(
            (
                (sq * valid).sum(1),
                (degrees * valid).sum(1),
                (width * valid).sum(1),
                valid.sum(1),
                degrees[:, 0],
                width[:, 0],
                degrees[:, -1] * valid[:, -1],
                width[:, -1] * valid[:, -1],
                valid[:, -1],
                (issued_width * valid).sum(1),
                (issued_width * movement).sum(1),
                movement.sum(1),
                ((pred - bank.normalized[ix]).abs().mean(-1) * valid).sum(1),
            ),
            1,
        )
        sums.index_add_(0, episodes[ix], values.double())
        if anchors:
            selected_sums.index_add_(
                0, episodes[ix], values[:, [12, 3]].double() * selection[ix, None]
            )
    if world > 1:
        dist.all_reduce(sums)
        if anchors:
            dist.all_reduce(selected_sums)
    numbers = sums.cpu().numpy()

    def metrics(totals, anchors):
        mse, joint, grip, labels, j0, g0, j11, g11, last, issued, moving_error, travel, l1 = totals
        result = dict(
            normalized_mse=mse / labels,
            normalized_l1=l1 / labels,
            joint_mae_deg=joint / labels,
            gripper_mae_mm=grip / labels,
            first_joint_deg=j0 / anchors,
            first_gripper_mm=g0 / anchors,
            last_joint_deg=j11 / last,
            last_gripper_mm=g11 / last,
            command_gripper_mae_mm=issued / labels,
        )
        if travel:
            result["moving_gripper_mae_mm"] = moving_error / travel
        return result

    all_metrics = metrics(numbers.sum(0), bank.sizes.sum())
    per = {
        s.group_id: metrics(n, size)
        for s, n, size in zip(bank.sources, numbers, bank.sizes, strict=True)
    }
    all_metrics["macro_normalized_mse"] = float(
        np.mean([v["normalized_mse"] for v in per.values()])
    )
    all_metrics["macro_normalized_l1"] = float(np.mean([v["normalized_l1"] for v in per.values()]))
    if anchors:
        all_metrics["selection_normalized_l1"] = (
            (selected_sums[:, 0] / selected_sums[:, 1]).mean().item()
        )
    subsets = {s.suite for s in bank.sources}
    for name in sorted(subsets):
        select = np.array([s.suite == name for s in bank.sources])
        group = metrics(numbers[select].sum(0), bank.sizes[select].sum())
        for key in ("joint_mae_deg", "gripper_mae_mm", "normalized_mse", "command_gripper_mae_mm"):
            all_metrics[f"{name}/{key}"] = group[key]
    return all_metrics, per


def command_targets(decoded):
    """The gripper width is physical finger separation, with zero at closed contact."""
    targets = np.array(decoded, copy=True)
    targets[..., 7] = np.maximum(targets[..., 7], 0)
    return targets


@torch.inference_mode()
def predict_actions(model, raw, state, *, history=None, return_scene=False):
    """Return eight decoded model outputs in radians and meters, including signed aperture."""
    cfg = model.cfg
    device = next(model.parameters()).device
    physical = np.asarray(state, dtype=np.float32).reshape(-1, 8)
    raws = np.asarray(raw)
    if raws.ndim == 3:
        raws = raws[None]
    if (
        raws.shape != (len(physical), 2, *cfg["observation"]["raw_shape"])
        or raws.dtype != np.uint16
    ):
        raise ValueError("Expected paired native uint16 RAW frames in U2, U3 order")

    def tensor(v):
        return torch.as_tensor(v, device=device)

    batch = dict(
        raw=tensor(raws.astype(np.float32)),
        state=tensor(normalize_state(physical, cfg["statistics"])),
        tokens=tensor(cfg["task_token_ids"]).expand(len(physical), -1),
        token_mask=tensor(cfg["task_token_mask"]).bool().expand(len(physical), -1),
    )
    if history is not None:
        past = np.asarray(history, dtype=np.float32)
        if past.shape != (len(physical), len(cfg["model"]["state_history_lags"]), 8):
            raise ValueError("History does not match the checkpoint's joint-state sampling times")
        batch["state_history"] = tensor(normalize_state(past, cfg["statistics"]))
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        if return_scene:
            y, scene = model.predict(batch, return_scene=True)
        else:
            y = model.sample(batch)
    actions = decode_actions(
        unnormalize_actions(y.float().cpu().numpy(), cfg["statistics"]), physical, cfg
    )
    return (actions, scene.float().cpu().numpy()) if return_scene else actions


def predict_commands(model, raw, state):
    """Use the physical closed-contact convention for command evaluation and historical replays."""
    return command_targets(predict_actions(model, raw, state))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--inventory", type=Path, help="Override the inventory saved at training time"
    )
    parser.add_argument("--episode", required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    ck = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    model = make_policy(ck["config"]).to(args.device).eval()
    model.load_state_dict(ck["model"], strict=True)
    sources, _ = load_inventory(args.inventory or ck["config"]["data"]["inventory"])
    src = next(s for s in sources if s.group_id == args.episode)
    a, b = raw_views(src)
    with np.load(src.path) as z:
        state = z["state"][args.frame]
        history = None
        lags = ck["config"]["model"].get("state_history_lags")
        if lags:
            from owac.runtime.teleop_history import causal_history_indices

            clock = z["observation_ns"]
            times = (clock - clock[0]) / 1e9
            rows = causal_history_indices(times, lags, 1 / 30)[args.frame]
            history = z["state"][rows][None]
    actions = predict_actions(
        model, np.stack((a[args.frame], b[args.frame])), state, history=history
    )
    y = command_targets(actions)[0]
    print(
        json.dumps(
            dict(
                step=ck["step"],
                episode=args.episode,
                frame=args.frame,
                units=["rad"] * 7 + ["m"],
                commands=y.tolist(),
            )
        )
    )


if __name__ == "__main__":
    main()
