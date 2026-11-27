"""Fit a U2 banana patch readout on frozen native OWAC features and reviewed weak labels."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from owac.frameworks.raw_compact.factory import make_policy
from owac.raw.all_bench_distillation import load_inventory
from owac.training.fruit_grasp import banana_image_errors, grasp_phase
from owac.training.raw_ssl_train import atomic_save
from owac.training.teleop_policy_data import FruitFrames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--balance-approach", action="store_true")
    parser.add_argument("--run-id", default="fruit-night19-banana-localizer")
    parser.add_argument("--name", default="Fruit | 香蕉定位预训练")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import swanlab

    torch.set_num_threads(2)
    seed = 20261120
    torch.manual_seed(seed)
    parent = torch.load(args.parent, map_location="cpu", mmap=True, weights_only=False)
    cfg = parent["config"]
    cfg["model"].update(spatial_goal_head={"width": 128, "arms": 1}, banana_grounding=True)
    cfg["data"]["banana_labels"] = str(args.labels)
    model = make_policy(cfg).cuda().eval()
    weights = model.state_dict()
    weights.update(parent["model"])
    model.load_state_dict(weights, strict=True)
    sources, _ = load_inventory(cfg["data"]["inventory"])
    train, val = (
        FruitFrames([s for s in sources if s.split == split], cfg, torch.device("cuda"))
        for split in ("train", "validation")
    )
    for bank in (train, val):
        bank.load_views("features")
    model.requires_grad_(False)
    head = model.spatial_goal_head.requires_grad_(True)
    with torch.no_grad():
        language = model.conditioning(train.batch(train.tensor([0])))[1]
    opt = torch.optim.AdamW(head.parameters(), lr=0.001)
    ids = []
    for start, n in zip(train.starts, train.sizes, strict=True):
        ids.append(
            np.flatnonzero(train.banana_visible[start : start + n].cpu().numpy() > 0) + start
        )
    approach_ids = (
        [
            start + np.flatnonzero(grasp_phase(src, 1.5)[1])
            for start, src in zip(train.starts, train.sources, strict=True)
        ]
        if args.balance_approach
        else []
    )
    mean = train.banana_mass[train.tensor(train.starts)].float().mean(0)
    val_ix = val.tensor(val.starts)
    occluded_ix = (
        val.tensor(
            [
                start + int(np.abs(grasp_phase(src, 1.5)[0] - 0.35).argmin())
                for start, src in zip(val.starts, val.sources, strict=True)
            ]
        )
        if args.balance_approach
        else None
    )
    grid = (30, 40)

    def readout(bank, ix):
        tokens = bank.image_bank[ix, 0].float()
        return head(tokens, language.expand(len(ix), -1))[:, 0]

    settings = dict(
        seed=seed,
        updates=2000,
        batch=32,
        parent_step=parent["step"],
        training_episodes=[s.group_id for s in train.sources],
        validation_episodes=[s.group_id for s in val.sources],
        reviewed_masks=str(args.labels),
        balance_approach=args.balance_approach,
    )
    run = swanlab.init(
        workspace=os.environ.get("SWANLAB_WORKSPACE"),
        mode=os.environ.get("SWANLAB_MODE", "local"),
        project=os.environ.get("SWANLAB_PROJECT", "OWAC"),
        id=args.run_id,
        resume="allow",
        name=args.name,
        group="real-fruit-overnight-20261119",
        job_type="native-raw-localization",
        public=False,
        config=settings,
        logdir=str(args.output / "swanlab"),
    )
    rng = np.random.default_rng(seed)
    rows = []
    for step in range(settings["updates"] + 1):
        if step:
            episodes = rng.integers(len(ids), size=settings["batch"])
            ix = train.tensor(
                [
                    rng.choice(approach_ids[i] if approach_ids and j % 2 else ids[i])
                    for j, i in enumerate(episodes)
                ]
            )
            logits = readout(train, ix)
            ce = -(train.banana_mass[ix].float() * logits.log_softmax(-1)).sum(-1)
            visible = train.banana_visible[ix]
            loss = (ce * visible).sum() / visible.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if step % 100 == 0:
            with torch.no_grad():
                probs = readout(val, val_ix).softmax(-1)
                ref = val.banana_mass[val_ix]
                score = banana_image_errors(probs, ref, grid, 16)
                baseline = banana_image_errors(mean.expand_as(ref), ref, grid, 16)
            values = {
                **{"validation/" + k: v for k, v in score.items()},
                **{"baseline/train_mean_" + k: v for k, v in baseline.items()},
            }
            if occluded_ix is not None:
                with torch.no_grad():
                    s = banana_image_errors(readout(val, occluded_ix).softmax(-1), ref, grid, 16)
                values.update({"before_close/" + k: v for k, v in s.items()})
            if step:
                values["train/heatmap_cross_entropy"] = loss.item()
            rows.append(dict(step=step, metrics=values))
            swanlab.log(values, step=step)
    atomic_save(dict(head=head.state_dict(), settings=settings, step=step), args.output / "head.pt")
    (args.output / "scores.json").write_text(
        json.dumps(dict(url=run.url, history=rows), indent=2) + "\n"
    )
    print(json.dumps(rows[-1]), flush=True)
    swanlab.finish()


if __name__ == "__main__":
    main()
