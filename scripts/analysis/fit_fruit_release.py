"""Fit a conditional aperture regressor from the demonstrated holding and opening strokes."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from evaluate_fruit_placement import opening_times
from torch.nn import functional as F

from owac.frameworks.raw_compact.fruit_placement import PlacementGate, ReleaseAperture
from owac.raw.all_bench_distillation import load_inventory
from owac.training.fruit_grasp import NeroHand
from owac.training.fruit_placement import placement_events
from owac.training.teleop_policy_data import episode_labels


def opening_distance(offsets, event, measured_xy):
    """Distance from demonstrated release along the observed path for each replan offset."""

    def at(t):
        return np.array([np.interp(t, event.time, axis) for axis in measured_xy.T])

    goal = at(event.release)
    return [
        float(np.linalg.norm(at(event.close if v is None else event.release + v) - goal))
        for v in offsets
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "gate", "descriptors", "original-replay", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20261120)
    torch.set_num_threads(2)
    rng = np.random.default_rng(20261120)
    cfg = torch.load(args.checkpoint, mmap=True, map_location="cpu", weights_only=False)["config"]
    sources = load_inventory(cfg["data"]["inventory"])[0]
    train = [s for s in sources if s.split == "train"]
    timings = {s.group_id: placement_events(s) for s in sources}
    hand = NeroHand(cfg["grasp_evaluation"]["urdf"], "cuda")
    aperture = ReleaseAperture(cfg, len(train)).cuda()
    for k in ("xyz", "rot", "tip", "basis"):
        getattr(aperture, k).copy_(getattr(hand, k))
    q_release = np.stack([timings[s.group_id].release_command for s in train])
    aperture.releases.copy_(hand.pose(torch.from_numpy(q_release).cuda())[0])
    gate = PlacementGate(cfg["model"]["raw_encoder"]["width"]).cuda().eval()
    gate.load_state_dict(torch.load(args.gate, map_location="cpu", weights_only=True))
    X, G, Y, B, V, pools, validation = [], [], [], [], [], [], []
    lo = cfg["statistics"]["action_q01"][7]
    span = cfg["statistics"]["action_q99"][7] - lo + 1e-6
    hz = cfg["observation"]["frequency_hz"]
    start = 0
    for src in sources:
        e = timings[src.group_id]
        state, actions, _, valid = episode_labels(src, cfg)
        n, horizon = actions.shape[:2]
        current = torch.from_numpy(state).cuda()
        inp = torch.load(
            args.descriptors / (src.group_id + ".pt"), map_location="cuda", weights_only=True
        )
        geom = aperture.geometry_features(inp[:, -8:])
        width = actions[..., 7]
        after = (e.time >= e.grasp_complete) & (e.time <= e.opening_end)
        # The midpoint of this fruit's release stroke separates an open command from a hold.
        low, high = width[after & valid[:, 0], 0].min(), width[after & valid[:, 0], 0].max()
        X.append((inp - gate.mean) / gate.scale)
        G.append(geom)
        Y.append(torch.as_tensor((width - lo) * 2 / span - 1, device="cuda"))
        B.append(torch.as_tensor(width > (low + high) / 2, device="cuda"))
        V.append(torch.as_tensor(valid, device="cuda"))
        if src.split == "train":
            prepare = e.opening_start - (horizon - 1) / hz
            pools.append(
                [
                    start + np.flatnonzero(mask)
                    for mask in (
                        after & (e.time < prepare),
                        after & (e.time >= prepare) & (e.time <= e.peak),
                        after & (e.time > e.peak),
                    )
                ]
            )
        else:
            base = np.load(args.original_replay / (src.group_id.replace("/", "-") + ".npz"))[
                "predicted"
            ]
            routing = gate(inp).detach().argmax(-1).cpu().numpy().astype(bool)
            xy = hand.pose(current)[0].cpu().numpy()[:, :2] * 1000
            validation.append((src.group_id, slice(start, start + n), base, routing, valid, e, xy))
        start += n
    x, g, y, label, valid_all = map(torch.cat, (X, G, Y, B, V))
    indices = np.concatenate([a for group in pools for a in group])
    aperture.geom_mean.copy_(g[indices].mean(0))
    aperture.geom_scale.copy_(g[indices].std(0).clamp_min(1e-6))
    opt = torch.optim.AdamW(aperture.parameters(), lr=0.0005)
    best = np.inf
    for step in range(1, 1501):
        ep, stage = rng.integers(len(pools), size=128), rng.integers(3, size=128)
        ix = [rng.choice(pools[i][j]) for i, j in zip(ep, stage, strict=True)]
        logits, holding, opening = aperture(x[ix], g[ix])
        use = valid_all[ix]
        classification = F.binary_cross_entropy_with_logits(logits[use], label[ix][use].float())
        value = torch.where(label[ix], opening, holding)
        loss = classification + F.mse_loss(value[use], y[ix][use])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 100:
            continue
        episodes = {}
        with torch.no_grad():
            for eid, sl, base, route, valid, e, xy in validation:
                v = aperture.predict(x[sl], g[sl]).cpu().numpy()
                pred = base.copy()
                pred[route, :, 7] = (v[route] + 1) * span / 2 + lo
                offsets = opening_times(pred, valid, e, hz)
                distances = opening_distance(offsets, e, xy)
                episodes[eid] = dict(misses=offsets.count(None), offsets_s=offsets, xy_mm=distances)
        score = np.mean([np.mean(a["xy_mm"]) for a in episodes.values()])
        if score < best:
            best = score
            torch.save(aperture.state_dict(), args.output / "aperture.pt")
            (args.output / "validation.json").write_text(
                json.dumps(
                    dict(
                        step=step, event_distance_mm=score, episodes=episodes, landmarks=len(train)
                    ),
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
