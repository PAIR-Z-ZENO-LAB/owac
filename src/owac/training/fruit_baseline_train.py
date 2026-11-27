"""Train fruit baselines and replay selected checkpoints on held-out episodes."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from owac.raw.all_bench_distillation import load_inventory
from owac.training.fruit_baseline_models import (
    export,
    load_export,
    make_model,
    rawvla_offsets,
    weights,
)
from owac.training.fruit_grasp import NeroHand
from owac.training.fruit_placement import episode_scores, opening_samples, placement_events
from owac.training.teleop_policy_data import FruitFrames, fit_statistics


class FruitInputs(FruitFrames):
    def __init__(self, sources, cfg, method, device):
        super().__init__(sources, cfg, device)
        self.method = method
        key = "dino224" if method == "dino-corp" else "linear"
        for src, start, n in zip(sources, self.starts, self.sizes, strict=True):
            path = Path(cfg["root"]) / "cache" / src.group_id
            receipt = json.loads((path / "inputs.json").read_text())
            if (
                receipt["episode"] != src.group_id
                or receipt["revision"] != cfg["data"]["dataset_revision"]
            ):
                raise ValueError(f"Unexpected cached episode {src.group_id}")
            array = np.load(path / (key + ".npy"), mmap_mode="r")
            if len(array) != n:
                raise ValueError(f"Misaligned image count for {src.group_id}")
            if self.image_bank is None:
                self.image_bank = torch.empty(
                    (len(self.state), *array.shape[1:]), dtype=torch.float16, device=device
                )
            self.image_bank[start : start + n].copy_(torch.from_numpy(np.array(array)))
        self.episode_start = self.tensor(self.starts[self.episode])
        self.history = self.tensor(rawvla_offsets(cfg["pi05"]["recurrent_stride"]))
        stats = cfg["statistics"]
        lo, hi = (self.state.new_tensor(stats[k]) for k in ("state_q01", "state_q99"))
        self.state_q = (2 * (self.state - lo) / (hi - lo + 1e-6) - 1).clamp(-1, 1)

    def observation(self, ix):
        if self.method == "rawvla":
            history = torch.maximum(
                ix[:, None, None] + self.history, self.episode_start[ix, None, None]
            )
            images = self.image_bank[history].permute(0, 3, 1, 2, 4, 5, 6)
        else:
            images = self.image_bank[ix]
        return {"images": images, "state": self.state_norm[ix], "state_q": self.state_q[ix]}

    def supervised(self, ix):
        return dict(self.observation(ix), actions=self.normalized[ix], valid=self.valid[ix])


def save(payload, path):
    temp = path.with_suffix(".partial")
    torch.save(payload, temp)
    os.replace(temp, path)


def report(folder, step, values):
    values = {k: float(v) for k, v in values.items()}
    with (folder / "metrics.jsonl").open("a") as out:
        out.write(json.dumps(dict(step=step, values=values), allow_nan=False) + "\n")
    print(step, values, flush=True)


@torch.inference_mode()
def evaluate(model, bank, cfg, anchors, batch_size, folder=None):
    model.eval()
    generator = torch.Generator(device=bank.device).manual_seed(cfg["seed"] + 1)
    # Identical initial flow noise for each validation anchor across calls and both π models.
    noise = torch.randn(
        len(bank.state),
        cfg["model"]["action_horizon"],
        model.noise_dim,
        device=bank.device,
        generator=generator,
    )
    use_events = cfg.get("replay_protocol", "fruit_placement") == "fruit_placement"
    hand = NeroHand(cfg["assets"]["urdf"], "cpu") if folder and use_events else None
    scores = {}
    for src, start, n in zip(bank.sources, bank.starts, bank.sizes, strict=True):
        rows = (
            np.arange(n)
            if anchors is None
            else np.linspace(0, n - 1, min(anchors, n), dtype=np.int64)
        )
        ix = bank.tensor(rows + start)
        predictions = []
        for part in ix.split(batch_size):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predictions.append(model.sample(bank.observation(part), noise[part]))
        normalized = torch.cat(predictions).float()
        valid = bank.valid[ix]
        absolute = bank.physical(normalized, ix)
        row = {
            "normalized_l1": (normalized - bank.normalized[ix]).abs().mean(-1)[valid].mean().item()
        }
        if not use_events:
            error = (absolute - bank.absolute[ix]).abs()[valid]
            row["joint_mae_deg"] = error[:, :7].rad2deg().mean().item()
            row["gripper_mae_mm"] = (error[:, 7].mean() * 1000).item()
            if folder:
                np.savez_compressed(
                    folder / (src.group_id.replace("/", "-") + ".npz"),
                    predicted=absolute.cpu().numpy(),
                    target=bank.absolute[ix].cpu().numpy(),
                    state=bank.state[ix].cpu().numpy(),
                    valid=valid.cpu().numpy(),
                )
            scores[src.group_id] = row
            continue
        if folder:
            q, target, state, valid_cpu = (
                x.cpu() for x in (absolute, bank.absolute[ix], bank.state[ix], valid)
            )
            event = placement_events(src)
            row.update(
                episode_scores(
                    hand, q, target, state, valid_cpu, event, cfg["observation"]["frequency_hz"]
                )
            )
            opening = opening_samples(
                q.numpy(), valid_cpu.numpy(), event, cfg["observation"]["frequency_hz"]
            )
            pos = hand.pose(torch.from_numpy(event.release_command))[0].numpy()[:2]
            xy = hand.pose(q)[0].numpy()[..., :2]
            row["first_open_minus_expert_s"] = [r[0] if r else None for r in opening]
            row["first_open_xy_error_mm"] = [
                float(np.linalg.norm(xy[r[1]] - pos) * 1000) if r else None for r in opening
            ]
            np.savez_compressed(
                folder / (src.group_id.replace("/", "-") + ".npz"),
                predicted=q.numpy(),
                target=target.numpy(),
                state=state.numpy(),
                valid=valid_cpu.numpy(),
                time=event.time,
            )
        scores[src.group_id] = row
    return scores


def train(cfg, method, eval_only):
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    folder = Path(cfg["root"]) / "runs" / method
    folder.mkdir(parents=True, exist_ok=True)
    inv = Path(cfg["data"]["inventory"])
    if hashlib.sha256(inv.read_bytes()).hexdigest() != cfg["data"]["inventory_sha256"]:
        raise ValueError("Inventory revision mismatch")
    sources = load_inventory(inv)[0]
    training = [s for s in sources if s.split == "train"]
    validation = [s for s in sources if s.split == "validation"]
    if [s.group_id for s in validation] != cfg["validation"] or set(
        s.group_id for s in training
    ) & set(cfg["validation"]):
        raise ValueError("Unexpected fruit split")
    if (folder / "config.json").exists():
        existing = json.loads((folder / "config.json").read_text())
        if any(existing[k] != cfg[k] for k in cfg):
            raise ValueError("Existing run configuration differs")
        cfg = existing
    else:
        cfg["statistics"] = fit_statistics(cfg, training)
        states = []
        for src in training:
            with np.load(src.path) as z:
                states.append(z["state"])
        lo, hi = np.quantile(np.concatenate(states), [0.01, 0.99], axis=0)
        cfg["statistics"].update(state_q01=lo.tolist(), state_q99=hi.tolist())
        cfg.update(task_token_ids=[0], task_token_mask=[True])
        (folder / "config.json").write_text(json.dumps(cfg, indent=2))
    options = cfg["training" if method == "dino-corp" else "pi05"]
    steps, batch = options["updates"], options["batch"]
    micro = options.get("microbatch", batch)
    if batch % micro:
        raise ValueError("Incompatible effective and micro batch sizes")
    model = make_model(cfg, method).cuda()
    val = FruitInputs(validation, cfg, method, "cuda")
    params = weights(model)
    if not eval_only:
        bank = FruitInputs(training, cfg, method, "cuda")
        groups = [
            {
                "params": [p for n, p in params.items() if not n.startswith("frontend.")],
                "lr": options["learning_rate"],
            }
        ]
        if method != "dino-corp":
            groups.append(
                {"params": list(model.frontend.parameters()), "lr": options["frontend_lr"]}
            )
        optimizer = torch.optim.AdamW(
            groups, weight_decay=cfg["training"]["weight_decay"], fused=True
        )
        base_lrs = [g["lr"] for g in groups]
        last_step, best_score = 0, float("inf")
        if (folder / "resume.pt").exists():
            ck = torch.load(folder / "resume.pt", weights_only=False, map_location="cpu")
            load_export(model, ck["weights"])
            optimizer.load_state_dict(ck["optimizer"])
            last_step, best_score = ck["step"], ck["best"]
            torch.cuda.set_rng_state(ck["cuda_rng"])
            torch.set_rng_state(ck["cpu_rng"])
            del ck
        if last_step == 0 and cfg.get("replay_protocol") == "commands":
            initial = evaluate(
                model,
                val,
                cfg,
                options["selection_anchors_per_episode"],
                24 if method == "dino-corp" else 2,
            )
            report(folder, 0, command_curves(initial, "validation"))
        print(
            "trainable",
            sum(p.numel() for p in params.values()),
            "train frames",
            len(bank.state),
            "validation frames",
            len(val.state),
            flush=True,
        )
        for step in range(last_step + 1, steps + 1):
            model.train()
            warmup = min(cfg["training"]["warmup"], max(1, steps // 20))
            factor = (
                min(1.0, step / warmup)
                if step <= warmup
                else 0.1 + 0.9 * (1 + math.cos(math.pi * (step - warmup) / (steps - warmup))) / 2
            )
            for g, lr in zip(optimizer.param_groups, base_lrs, strict=True):
                g["lr"] = lr * factor
            optimizer.zero_grad(set_to_none=True)
            losses = {}
            ix = bank.sample_indices(cfg["seed"] + step, batch)
            for part in ix.split(micro):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, metrics = model.loss(bank.supervised(part))
                (loss * (micro / batch)).backward()
                for name, value in dict(metrics, total=loss.detach()).items():
                    losses[name] = losses.get(name, 0.0) + value.detach() * (micro / batch)
            norm = torch.nn.utils.clip_grad_norm_(
                list(params.values()), 1.0, error_if_nonfinite=True
            )
            if step == 1:
                for group in ("frontend.", "policy.", "resampler.", "generator."):
                    subset = [
                        p.grad.detach().norm()
                        for n, p in params.items()
                        if n.startswith(group) and p.grad is not None
                    ]
                    if subset:
                        losses["grad_" + group[:-1]] = torch.stack(subset).norm()
            optimizer.step()
            if step == 1 or step % (20 if method == "dino-corp" else 8) == 0:
                report(
                    folder,
                    step,
                    dict(
                        {"train/" + k: v for k, v in losses.items()},
                        **{"train/grad_norm": norm, "train/lr": optimizer.param_groups[0]["lr"]},
                    ),
                )
            if step % options["eval_every"] and step != steps:
                continue
            rows = evaluate(
                model,
                val,
                cfg,
                options["selection_anchors_per_episode"],
                24 if method == "dino-corp" else 2,
            )
            score = float(np.mean([v["normalized_l1"] for v in rows.values()]))
            validation_metrics = {"validation/normalized_l1": score}
            if cfg.get("replay_protocol") == "commands":
                validation_metrics.update(command_curves(rows, "validation"))
            report(folder, step, validation_metrics)
            payload = dict(method=method, step=step, config=cfg, weights=export(model), score=score)
            if score < best_score:
                best_score = score
                save(payload, folder / "best.pt")
            if step == steps:
                save(payload, folder / "final.pt")
            save(
                dict(
                    payload,
                    optimizer=optimizer.state_dict(),
                    best=best_score,
                    cuda_rng=torch.cuda.get_rng_state(),
                    cpu_rng=torch.get_rng_state(),
                ),
                folder / "resume.pt",
            )
        del bank, optimizer
    ck = torch.load(folder / "best.pt", weights_only=False, map_location="cpu")
    load_export(model, ck["weights"])
    replay = folder / "replay"
    replay.mkdir(exist_ok=True)
    scores = evaluate(model, val, cfg, None, 32 if method == "dino-corp" else 4, replay)
    result = dict(method=method, step=ck["step"], validation=scores)
    (replay / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    if cfg.get("replay_protocol") == "commands":
        report(
            folder, steps, command_curves(scores, "replay") | {"replay/selected_step": ck["step"]}
        )
        return
    macro = {
        k: float(np.mean([v[k] for v in scores.values()]))
        for k in ("normalized_l1", "pickup_xyz_mm", "carry_xy_mm", "release_xy_mm")
    }
    report(folder, steps, {"replay/" + k: v for k, v in macro.items()})


def command_curves(episodes, prefix):
    """Keep each trajectory visible and give each one equal weight in the aggregate."""
    curves = {f"{prefix}/{ep}/{key}": v for ep, row in episodes.items() for key, v in row.items()}
    for key in next(iter(episodes.values())):
        curves[f"{prefix}/{key}"] = float(np.mean([row[key] for row in episodes.values()]))
    return curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=["dino-corp", "rawvla", "rawild"])
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    train(json.loads(args.config.read_text()), args.method, args.eval_only)


if __name__ == "__main__":
    main()
