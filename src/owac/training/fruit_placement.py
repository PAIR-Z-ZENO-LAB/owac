"""Transport and release labels from an expert's first fruit pick and place."""

from dataclasses import dataclass

import numpy as np
import torch

from owac.training.fruit_grasp import closing_stroke


@dataclass
class Placement:
    time: np.ndarray
    close: float
    grasp_complete: float
    release: float
    opening_start: float
    peak: float
    opening_end: float
    half_width: float
    release_command: np.ndarray


def placement_events(source):
    with np.load(source.path) as data:
        observed = data["observation_ns"]
        command_time = (data["command_ns"] - observed[0]) / 1e9
        commands = data["commands"]
    widths = commands[:, 7]
    close, half = closing_stroke(widths, source.group_id)
    rises = np.flatnonzero((widths[:-1] <= half) & (widths[1:] > half)) + 1
    after_grasp = rises[rises > close]
    if not len(after_grasp):
        raise ValueError(f"No release after grasp in {source.group_id}")
    release = after_grasp[0]
    closed = widths[close:release]
    minimum = np.flatnonzero(closed == closed.min()) + close
    falls = np.flatnonzero((widths[release:-1] > half) & (widths[release + 1 :] <= half))
    end = release + 1 + falls[0] if len(falls) else len(widths) - 1
    peak = release + np.argmax(widths[release : end + 1])
    return Placement(
        (observed - observed[0]) / 1e9,
        command_time[close],
        command_time[minimum[0]],
        command_time[release],
        command_time[minimum[-1]],
        command_time[peak],
        command_time[end],
        half,
        commands[release],
    )


def placement_rows(events, seconds):
    """Disjoint pickup, transport and opening windows for episode-balanced sampling."""
    t = events.time
    opening_start = max(events.close, events.release - seconds)
    return [
        np.flatnonzero((t >= events.close - seconds) & (t < events.close)),
        np.flatnonzero((t >= events.close) & (t < opening_start)),
        np.flatnonzero((t >= opening_start) & (t <= events.opening_end)),
    ]


def opening_samples(q, valid, e, hz):
    """Try every initial chunk offset on recorded inputs; detect the first post-grasp reopen."""
    result = []
    horizon = q.shape[1]
    for offset in range(horizon):
        rows, widths, indices = [], [], []
        i = offset
        while i < len(q) and e.time[i] <= e.opening_end:
            ticks = e.time[i] + np.arange(horizon) / hz
            rows.extend(ticks[valid[i]])
            widths.extend(q[i, valid[i], 7])
            indices.extend((i, k) for k in np.flatnonzero(valid[i]))
            i = np.searchsorted(e.time, e.time[i] + horizon / hz)
        t, w = np.asarray(rows), np.asarray(widths)
        crosses = (w[:-1] <= e.half_width) & (w[1:] > e.half_width)
        rises = np.flatnonzero(crosses & (t[1:] > e.close) & (t[1:] <= e.opening_end)) + 1
        result.append((float(t[rises[0]] - e.release), indices[rises[0]]) if len(rises) else None)
    return result


def episode_scores(hand, q, target, state, valid, event, hz):
    """Score forecast targets on recorded observations. All positions are in the arm base."""
    xyz = hand.pose(q)[0] * 1000
    expert = hand.pose(target)[0] * 1000
    current = hand.pose(state)[0] * 1000
    device = q.device
    t = torch.as_tensor(event.time, device=device, dtype=torch.float64)
    tick = t[:, None] + torch.arange(q.shape[1], device=device) / hz
    length = (xyz - expert).norm(dim=-1)
    xy = (xyz[..., :2] - expert[..., :2]).norm(dim=-1)
    wrist = 1000 * (q[..., 7] - target[..., 7])
    scores = {}
    for name, instant in (
        ("pickup", event.close),
        ("release", event.release),
        ("peak", event.peak),
    ):
        mask = (t[:, None] <= instant) & ((tick - instant).abs() <= 0.5 / hz) & valid
        if not mask.any():
            raise ValueError(f"No valid prediction covers the {name} stroke")
        scores.update(
            {
                name + "_forecasts": int(mask.sum()),
                name + "_xyz_mm": length[mask].mean().item(),
                name + "_xy_mm": xy[mask].mean().item(),
                name + "_width_mae_mm": wrist[mask].abs().mean().item(),
                name + "_predicted_width_mm": (1000 * q[..., 7][mask]).mean().item(),
                name + "_expert_width_mm": (1000 * target[..., 7][mask]).mean().item(),
            }
        )
        if name == "release":
            release_error = torch.hypot(length[mask], wrist[mask]).mean()
    carry = (t >= event.close) & (tick[:, -1] < event.release) & valid[:, -1]
    actual_move = expert[carry, -1, :2] - current[carry, :2]
    model_move = xyz[carry, -1, :2] - current[carry, :2]
    goal = hand.pose(torch.as_tensor(event.release_command, device=device))[0][:2] * 1000
    toward = torch.nn.functional.normalize(goal - current[carry, :2], dim=-1)
    scores.update(
        {
            "carry_anchors": int(carry.sum()),
            "carry_xyz_mm": length[carry, -1].mean().item(),
            "carry_xy_mm": xy[carry, -1].mean().item(),
            "carry_model_progress_mm": (model_move * toward).sum(-1).mean().item(),
            "carry_expert_progress_mm": (actual_move * toward).sum(-1).mean().item(),
            "carry_projected_fraction": (
                (model_move * actual_move).sum() / actual_move.square().sum()
            ).item(),
            "selection_mm": ((length[carry, -1].mean() + release_error) / 2).item(),
        }
    )
    return scores


@torch.inference_mode()
def evaluate_placement(model, bank, hand, events):
    hz = bank.cfg["observation"]["frequency_hz"]
    by_episode = {}
    for src, start, n, timing in zip(bank.sources, bank.starts, bank.sizes, events, strict=True):
        index = torch.arange(int(start), int(start + n), device=bank.device)
        outputs = []
        for ix in index.split(16):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model.sample(bank.batch(ix))
            outputs.append(bank.physical(pred.float(), ix))
        by_episode[src.group_id] = episode_scores(
            hand,
            torch.cat(outputs),
            bank.absolute[index],
            bank.state[index],
            bank.valid[index],
            timing,
            hz,
        )
    mean = {
        k: float(np.mean([row[k] for row in by_episode.values()]))
        for k in next(iter(by_episode.values()))
        if not k.endswith(("_anchors", "_forecasts"))
    }
    return mean, by_episode
