"""Episode-local training anchors and event measurements for the pouring recordings."""

import numpy as np
import torch

from owac.runtime.teleop_history import causal_history_indices
from owac.training.compact_data import normalize_state
from owac.training.teleop_policy_data import episode_labels


def opening_strokes(width, threshold):
    opened = np.r_[False, width > threshold, False]
    edges = np.flatnonzero(opened[1:] != opened[:-1]).reshape(-1, 2)
    if len(edges) < 2:
        return None
    longest = sorted(edges, key=lambda pair: pair[1] - pair[0])[-2:]
    return sorted(longest, key=lambda pair: pair[0])


def wrist_tilt(rotation, grasp):
    # The cup is assumed upright at grasp and rigidly attached during the pour.
    # Project its inferred vertical axis onto world vertical after wrist rotation.
    cosine = rotation[:, 2, :] @ rotation[grasp, 2, :]
    return np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))


class PouringBank:
    """Keep per-episode clocks and windows intact, including at both split boundaries."""

    def __init__(self, episodes, cache, cfg, hand):
        self.names = [s.group_id for s in episodes]
        self.rows = []
        self.pools = []
        fields = {k: [] for k in ("q", "past", "target", "valid", "grid", "base")}
        stat = cfg["statistics"]
        self.low = torch.tensor(stat["action_q01"], device="cuda")
        self.scale = torch.tensor(stat["action_q99"], device="cuda") - self.low + 1e-6
        self.mean = torch.tensor(stat["state_mean"], device="cuda")
        self.std = torch.tensor(stat["state_std"], device="cuda") + 1e-6
        self.mask = torch.tensor(cfg["observation"]["relative_action_mask"], device="cuda")
        offset = 0
        for src in episodes:
            with np.load(src.path) as data:
                times = data["observation_ns"].copy()
            q, target, _, valid = episode_labels(src, cfg)
            past = causal_history_indices(times, cfg["model"]["state_history_lags"], 1e9 / 30)
            delay = causal_history_indices(times, [0, 3, 6], 1e9 / 30)
            qnorm = normalize_state(q, stat)
            width = target[:, 0, 7]
            threshold = float(np.quantile(width, [0.1, 0.95]).mean())
            strokes = opening_strokes(width, threshold)
            if strokes is None:
                raise ValueError(f"Missing reference grasp and release in {src.group_id}")
            grasp, release = int(strokes[0][1]), int(strokes[1][0])
            _, rotation = hand.pose(torch.from_numpy(q))
            tilt = wrist_tilt(rotation.numpy(), grasp)
            carry = np.arange(grasp, release)
            tilted = carry[tilt[carry] >= tilt[carry].max() / 2]
            # One action horizon on either side of each command transition.
            radius = cfg["model"]["action_horizon"]
            events = np.arange(len(q))
            edges = np.unique(np.asarray(strokes).reshape(-1)).clip(0, len(q) - 1)
            near = (abs(events[:, None] - edges) <= radius).any(1)
            self.pools.append([offset + events, offset + events[near], offset + tilted])
            self.rows.append(
                dict(
                    name=src.group_id,
                    start=offset,
                    size=len(q),
                    times=times,
                    grasp=grasp,
                    release=release,
                    threshold=threshold,
                    tilted=tilted,
                )
            )
            offset += len(q)
            for key, value in (
                ("q", qnorm[delay]),
                ("past", qnorm[past[delay]]),
                ("target", target),
                ("valid", valid),
            ):
                fields[key].append(torch.from_numpy(value))
            with np.load(cache / (src.group_id + ".npz")) as saved:
                for key in ("grid", "base"):
                    fields[key].append(torch.from_numpy(saved[key]).view(torch.bfloat16))
        for key, arrays in fields.items():
            setattr(self, key, torch.cat(arrays).cuda())

    def indices(self, rng, size):
        # Half the batch covers the whole episode, one quarter each covers transitions and tilt.
        ix = []
        for episode, pool in zip(
            rng.integers(len(self.pools), size=size),
            rng.choice([0, 0, 1, 2], size=size),
            strict=True,
        ):
            ix.append(rng.choice(self.pools[episode][pool]))
        return torch.as_tensor(np.asarray(ix), device="cuda")

    def physical(self, normalized, q):
        anchor = (q * self.std + self.mean) * self.mask
        return (normalized.float() + 1) * (self.scale / 2) + self.low + anchor[:, None]


@torch.inference_mode()
def compare_actions(bank, model, delay):
    physical, original = [], []
    all_ix = torch.arange(len(bank.q), device="cuda")
    for ix in all_ix.split(128):
        q, past = bank.q[ix, delay], bank.past[ix, delay]
        base = bank.base[ix, delay]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            corrected = model.correct(base, bank.grid[ix], q, past)
        physical.append(bank.physical(corrected, q).cpu().numpy())
        original.append(bank.physical(base, q).cpu().numpy())
    return np.concatenate(original), np.concatenate(physical)


def score_actions(bank, actions, hand):
    target = bank.target.cpu().numpy()
    valid = bank.valid.cpu().numpy()
    details = []
    for ep in bank.rows:
        start, size = ep["start"], ep["size"]
        sl = slice(start, start + size)
        pred, truth, ok = actions[sl], target[sl], valid[sl]
        err = np.abs(pred - truth)
        events = np.array([ep["grasp"], ep["release"]])
        strokes = opening_strokes(pred[:, 0, 7].clip(0, 0.1), ep["threshold"])
        ref = ep["times"]
        n = len(ref)
        record = dict(
            name=ep["name"],
            joint_deg=float(np.rad2deg(err[:, :, :7][ok]).mean()),
            grip_mm=float(err[:, :, 7][ok].mean() * 1000),
            tilt_rows=len(ep["tilted"]),
            open_during_tilt=int((pred[ep["tilted"], 0, 7] > ep["threshold"]).sum()),
            paired_strokes=int(strokes is not None),
        )
        # The final row of the deployed four-command prefix reflects its arm trajectory.
        pos, rot = hand.squared_errors(
            torch.from_numpy(pred[events, 3]), torch.from_numpy(truth[events, 3])
        )
        record["event_hand_mm"] = float(pos.sqrt().mean() * 1000)
        record["event_rotation_deg"] = float(
            torch.rad2deg(torch.acos((1 - rot / 2).clamp(-1, 1))).mean()
        )
        if strokes is not None:
            chosen = np.array([min(strokes[0][1], n - 1), strokes[1][0]])
            grasp_dt, release_dt = (ref[chosen] - ref[events]) * 1e-6
            record.update(grasp_dt_ms=float(grasp_dt), release_dt_ms=float(release_dt))
        details.append(record)
    summary = {
        k: float(np.mean([row[k] for row in details]))
        for k in (
            "joint_deg",
            "grip_mm",
            "event_hand_mm",
            "event_rotation_deg",
        )
    }
    for key in ("tilt_rows", "open_during_tilt", "paired_strokes"):
        summary[key] = sum(row[key] for row in details)
    for key in ("grasp_dt_ms", "release_dt_ms"):
        observed = [abs(row[key]) for row in details if key in row]
        if observed:
            summary[key.replace("_dt_", "_abs_")] = float(np.mean(observed))
    return dict(summary=summary, episodes=details)
