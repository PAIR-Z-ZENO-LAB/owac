"""Fit a temporal residual covariance to native fruit regression outputs on training episodes."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from owac.frameworks.raw_compact.factory import make_policy
from owac.raw.all_bench_distillation import load_inventory
from owac.training.teleop_policy_data import FruitFrames
from owac.training.teleop_policy_eval import command_targets, evaluate


def residual_covariance(residuals):
    """Zero-mean Gaussian residuals per actuator with oracle-approximating shrinkage."""
    n, horizon, _ = residuals.shape
    data = residuals.astype(np.float64)
    cov = np.einsum("ntj,nsj->jts", data, data) / n
    mu = np.trace(cov, axis1=1, axis2=2) / horizon
    alpha = np.square(cov).mean(axis=(1, 2))
    denominator = (n + 1) * (alpha - mu**2 / horizon)
    ratio = np.divide(alpha + mu**2, denominator, out=np.ones_like(mu), where=denominator > 0)
    shrink = np.clip(ratio, 0, 1)
    cov *= (1 - shrink)[:, None, None]
    cov += (shrink * mu)[:, None, None] * np.eye(horizon)
    return cov


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--validation-traces", type=Path)
    args = p.parse_args()
    torch.set_num_threads(2)
    device = torch.device("cuda:0")
    ck = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    cfg = ck["config"]
    model = make_policy(cfg).to(device).eval()
    model.load_state_dict(ck["model"], strict=True)
    sources, _ = load_inventory(cfg["data"]["inventory"])
    with args.checkpoint.open("rb") as file:
        digest = hashlib.sha256()
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    for split in ("train", "validation") if args.validation_traces else ("train",):
        episodes = [s for s in sources if s.split == split]
        bank = FruitFrames(episodes, cfg, device)
        bank.load_views("raw")
        targets = bank.absolute.cpu().numpy()
        valid = bank.valid.cpu().numpy()
        raw = np.empty_like(targets)
        evaluate(model, bank, batch_size=16, predictions=raw)
        commands = command_targets(raw)
        if split == "train":
            complete = valid.all(axis=1)
            residuals = (targets - commands)[complete]
            payload = dict(
                version=1,
                step=ck["step"],
                checkpoint_sha256=digest.hexdigest(),
                dataset_revision=cfg["data"]["dataset_revision"],
                episodes=[s.group_id for s in episodes],
                anchors=len(complete),
                complete_chunks=int(complete.sum()),
                frequency_hz=cfg["observation"]["frequency_hz"],
                covariance=residual_covariance(residuals).tolist(),
            )
            args.output.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n")
            print(
                f"{ck['step']} fit {len(residuals)} chunks on {len(episodes)} train episodes",
                flush=True,
            )
        else:
            times = []
            for ep in episodes:
                with np.load(ep.path) as archive:
                    t = archive["observation_ns"]
                    times.append((t - t[0]) * 1e-9)
            np.savez_compressed(
                args.validation_traces,
                episode=bank.episode,
                time=np.concatenate(times),
                target=targets,
                valid=valid,
                predicted=commands,
                state=bank.state.cpu().numpy(),
            )
            print(
                f"{ck['step']} saved native predictions on {len(commands)} validation frames",
                flush=True,
            )
        del bank


if __name__ == "__main__":
    main()
