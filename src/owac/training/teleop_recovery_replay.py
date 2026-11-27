"""State-feedback replay using real views and a locally identified follower response."""

import numpy as np
import torch

from owac.training.teleop_policy_data import command_chunks
from owac.training.teleop_recovery import predict_cached


class RecoveryBank:
    def __init__(self, episodes, visual_dir, cfg, tracking):
        states, labels, masks, intervals, views = [], [], [], [], []
        self.clocks, self.thresholds, self.names = [], [], []
        self.low = self.tensor(tracking["low"])
        self.high = self.tensor(tracking["high"])
        self.tau = self.tensor([r["tau_s"] for r in tracking["servo"]])
        self.delay = self.tensor([r["delay_s"] for r in tracking["servo"]])
        horizon = cfg["model"]["action_horizon"]
        hz = cfg["observation"]["frequency_hz"]
        for ep in episodes:
            with np.load(ep.path) as z:
                clock, commands = z["observation_ns"], z["commands"]
                target, valid = command_chunks(clock, z["command_ns"], commands, horizon, hz)
                states.append(z["state"])
                labels.append(target)
                masks.append(valid)
                intervals.append(np.r_[np.diff(clock) / 1e9, 1 / hz])
                self.clocks.append((clock - clock[0]) / 1e9)
                self.thresholds.append(float(np.quantile(commands[:, 7], [0.05, 0.95]).mean()))
            self.names.append(ep.group_id)
            values = np.load(visual_dir / (ep.group_id + ".npy"), mmap_mode="r")
            if len(values) != len(clock):
                raise ValueError(f"Vision and state lengths differ for {ep.group_id}")
            views.append(torch.from_numpy(values.copy()).view(torch.bfloat16).cuda())
        self.visual = torch.cat(views)
        self.q = self.tensor(np.concatenate(states))
        self.target = self.tensor(np.concatenate(labels)).clamp(self.low, self.high)
        self.valid = torch.from_numpy(np.concatenate(masks)).cuda()
        self.dt = self.tensor(np.concatenate(intervals))
        self.sizes = np.array(list(map(len, states)))
        self.starts = np.r_[0, np.cumsum(self.sizes)[:-1]]
        self.start = torch.tensor(self.starts, device="cuda")
        self.size = torch.tensor(self.sizes, device="cuda")
        self.motion = []
        width = self.target[:, :, 7].cpu().numpy()
        for a, n in zip(self.starts, self.sizes, strict=True):
            variation = np.abs(np.diff(width[a : a + n], axis=1)).sum(1)
            self.motion.append(variation / variation.sum())
        stats = cfg["statistics"]
        self.mean = self.tensor(stats["state_mean"])
        self.sd = self.tensor(stats["state_std"]) + 1e-6

    @staticmethod
    def tensor(value):
        return torch.as_tensor(value, device="cuda", dtype=torch.float32)

    def indices(self, rng, batch):
        episode = rng.integers(len(self.sizes), size=batch)
        frames = (rng.random(batch) * self.sizes[episode]).astype(int)
        # Include both quiet motion and grip transitions anywhere in the two transfers.
        for i in range(0, batch, 2):
            e = episode[i]
            frames[i] = rng.choice(self.sizes[e], p=self.motion[e])
        return torch.as_tensor(self.starts[episode] + frames, device="cuda")

    def configure_output(self, cfg):
        stats = cfg["statistics"]
        self.zero = self.tensor(stats["action_q01"])
        self.half = (self.tensor(stats["action_q99"]) - self.zero + 1e-6) / 2
        self.relative = self.tensor(cfg["observation"]["relative_action_mask"])

    def load_spatial(self, directory):
        self.maps = torch.cat(
            [
                torch.from_numpy(np.load(directory / (name + ".npy"))).view(torch.bfloat16).cuda()
                for name in self.names
            ]
        )

    def history_rows(self, lags):
        from owac.frameworks.raw_compact.temporal_spatial_regression import causal_history_indices

        rows = np.concatenate(
            [
                start + causal_history_indices(clock, lags, 1 / 30)
                for start, clock in zip(self.starts, self.clocks, strict=True)
            ]
        )
        self.history_index = torch.as_tensor(rows, device="cuda")

    def predict(self, model, text, index, state, past=None):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            norm = predict_cached(model, self.visual[index], (state - self.mean) / self.sd, text)
            if hasattr(model, "visual_plan"):
                maps = self.maps[index]
                vision = model.visual_plan(maps)
                if hasattr(model, "correct"):
                    history = (past - self.mean) / self.sd if past is not None else None
                    norm = model.correct(norm, vision, maps, (state - self.mean) / self.sd, history)
                else:
                    norm = model.combine(
                        norm, vision, maps=maps, state=(state - self.mean) / self.sd
                    )
        return (norm.float() + 1) * self.half + self.zero + (state * self.relative)[:, None]

    @torch.no_grad()
    def aligned_rollout(self, model, text, age, ensemble):
        """Issue a target at each sensor tick using forecasts for that absolute time."""
        from collections import deque

        episodes = torch.arange(len(self.start), device="cuda")
        clock = torch.as_tensor(np.concatenate(self.clocks), device="cuda", dtype=torch.float64)
        previous = torch.zeros_like(self.q[self.start])
        delta = previous.clone()
        errors = torch.zeros_like(self.q)
        recent = deque(maxlen=12)
        for t in range(int(self.sizes.max())):
            ix = self.start + torch.minimum(torch.full_like(self.size, t), self.size - 1)
            active = self.size > t
            errors[ix[active]] = delta[active]
            past = None
            if hasattr(model, "correct"):
                rows = self.history_index[ix]
                past = self.q[rows] + errors[rows]
            recent.append((t, clock[ix], self.predict(model, text, ix, self.q[ix] + delta, past)))
            values = []
            valid = []
            for frame, stamp, plan in reversed(recent):
                if t - frame < age:
                    continue
                row = torch.floor((clock[ix] - stamp) * 30 + 0.5).long()
                valid.append((row >= 0) & (row < plan.shape[1]))
                values.append(plan[episodes, row.clamp(0, plan.shape[1] - 1)])
                if not ensemble:
                    break
            if values:
                mask = torch.stack(valid)
                summed = (torch.stack(values) * mask[:, :, None]).sum(0)
                command = torch.where(
                    mask.any(0)[:, None],
                    summed / mask.sum(0).clamp_min(1)[:, None],
                    self.q[ix] + delta,
                ).clamp(self.low, self.high)
            else:
                command = self.q[ix] + delta
            current = command - self.target[ix, 0]
            dt = self.dt[ix, None]
            a = torch.exp(-torch.minimum(dt, self.delay) / self.tau)
            b = torch.exp(-(dt - self.delay).clamp_min(0) / self.tau)
            delta = b * (a * delta + (1 - a) * previous) + (1 - b) * current
            ref = self.q[
                self.start + torch.minimum(torch.full_like(self.size, t + 1), self.size - 1)
            ]
            delta = (ref + delta).clamp(
                torch.minimum(ref, self.low), torch.maximum(ref, self.high)
            ) - ref
            previous = current
        return errors

    @torch.no_grad()
    def rollout(self, model, text, send, *, gaps=(1,), postprocess=None):
        """Propagate command deviations around the measured expert trajectory.

        The field console waited a median of two camera periods between its last row and
        the next first row. Sample one fresh observation and hold the previous command for
        that tick. The same recorded images are used as the predicted state departs.
        """
        delta = torch.zeros_like(self.q[self.start])
        previous = delta.clone()
        command = self.target[self.start, 0]
        errors = torch.zeros_like(self.q)
        t = 0
        chunk = 0
        while t < self.sizes.max():
            gap = gaps[chunk % len(gaps)]
            ix = self.start + torch.minimum(torch.full_like(self.size, t), self.size - 1)
            past = None
            if hasattr(model, "correct"):
                rows = self.history_index[ix]
                # The latest measured sample may also be the padded startup history.
                past = self.q[rows] + torch.where(
                    (rows == ix[:, None])[:, :, None], delta[:, None], errors[rows]
                )
            raw = self.predict(model, text, ix, self.q[ix] + delta, past)
            plan = raw.clamp(self.low, self.high)
            if postprocess is not None:
                raw_chunk = raw[:, :send].cpu().numpy()
            for row in range(send + gap):
                frame = t + row
                ix = self.start + torch.minimum(torch.full_like(self.size, frame), self.size - 1)
                active = self.size > frame
                errors[ix[active]] = delta[active]
                if row >= gap:
                    command = plan[:, row - gap]
                    if postprocess is not None:
                        values = postprocess(
                            raw_chunk, row - gap, t, frame, (self.q[ix] + delta).cpu().numpy()
                        )
                        command = self.tensor(values).clamp(self.low, self.high)
                current = command - self.target[ix, 0]
                dt = self.dt[ix, None]
                a = torch.exp(-torch.minimum(dt, self.delay) / self.tau)
                b = torch.exp(-(dt - self.delay).clamp_min(0) / self.tau)
                delta = b * (a * delta + (1 - a) * previous) + (1 - b) * current
                next_ix = self.start + torch.minimum(
                    torch.full_like(self.size, frame + 1), self.size - 1
                )
                ref = self.q[next_ix]
                delta = (ref + delta).clamp(
                    torch.minimum(ref, self.low), torch.maximum(ref, self.high)
                ) - ref
                previous = current
            t += send + gap
            chunk += 1
        return errors


def event_points(time, q, threshold):
    """Return the first two picks and releases in order, using interpolated measured width."""
    opened = q[:, 7] > threshold
    changes = np.flatnonzero(opened[1:] != opened[:-1]) + 1
    events = []
    for i in changes:
        if not events and opened[i]:
            continue
        frac = (threshold - q[i - 1, 7]) / (q[i, 7] - q[i - 1, 7])
        events.append(
            (
                float(time[i - 1] + frac * (time[i] - time[i - 1])),
                q[i - 1] + frac * (q[i] - q[i - 1]),
            )
        )
        if len(events) == 4:
            break
    return events


def score_replay(bank, errors, hand):
    actual = bank.q.cpu().numpy()
    simulated = (bank.q + errors).cpu().numpy()

    def positions(q):
        return hand.pose(torch.from_numpy(np.asarray(q)[..., :7]))[0].numpy() * 1000

    rows = {}
    for name, a, n, clock, width in zip(
        bank.names, bank.starts, bank.sizes, bank.clocks, bank.thresholds, strict=True
    ):
        ref, sim = actual[a : a + n], simulated[a : a + n]
        expected, predicted = event_points(clock, ref, width), event_points(clock, sim, width)
        phase = []
        for (tt, qt), (tp, qp) in zip(expected, predicted, strict=False):
            diff = positions(qp) - positions(qt)
            phase.append(
                dict(
                    xy_mm=float(np.linalg.norm(diff[:2])),
                    xyz_mm=float(np.linalg.norm(diff)),
                    timing_s=float(tp - tt),
                    delta_xyz_mm=diff.tolist(),
                )
            )
        full = positions(sim) - positions(ref)
        rows[name] = dict(
            events=phase,
            expected=len(expected),
            full_hand_mm=float(np.mean(np.linalg.norm(full, axis=-1))),
            full_gripper_mm=float(1000 * np.abs(sim[:, 7] - ref[:, 7]).mean()),
        )
    all_events = [e for row in rows.values() for e in row["events"]]
    first = [r["events"][0] for r in rows.values() if r["events"]]
    summary = dict(
        events=len(all_events),
        expected=sum(r["expected"] for r in rows.values()),
        first_grasp_xy_mm=float(np.mean([e["xy_mm"] for e in first])) if first else None,
        four_events_xyz_mm=float(np.mean([e["xyz_mm"] for e in all_events]))
        if all_events
        else None,
        four_events_xy_mm=float(np.mean([e["xy_mm"] for e in all_events])) if all_events else None,
        four_events_timing_mae_s=float(np.mean([abs(e["timing_s"]) for e in all_events]))
        if all_events
        else None,
        full_hand_mm=float(np.mean([r["full_hand_mm"] for r in rows.values()])),
        full_gripper_mm=float(np.mean([r["full_gripper_mm"] for r in rows.values()])),
    )
    return summary, rows
