"""Time-aligned stereo RAW, follower state and applied command samples, by episode."""

import numpy as np
from torch.utils.data import Dataset

from owac.raw.all_bench_distillation import read_npz_member
from owac.raw.teleop_mcap import CAMERAS


def sample_indices(observation, state_time, action_time, commands, future_ns, period_ns):
    """Current and future labels are the first commands at/after their requested time."""
    if future_ns < 0 or period_ns < 0:
        raise ValueError("Time offsets must be nonnegative")
    target_time = observation + future_ns
    target = np.searchsorted(commands, target_time)
    rows = np.flatnonzero(
        (target < len(commands)) & (state_time <= observation) & (action_time >= observation)
    )
    if period_ns and len(rows):
        # First valid row of each occupied time bin. A gap cannot duplicate an anchor.
        bins = (observation[rows] - observation[0]) // period_ns
        rows = rows[np.r_[True, np.diff(bins) > 0]]
    return rows, target[rows]


class TeleopFrames(Dataset):
    """Read original uint16 [2,H,W] in U2,U3 order and physical 7 rad + 1 m vectors.

    action is the first command at/after observation; future_action is a label only.
    Each episode is indexed separately. The same item cannot straddle trajectories.
    """

    def __init__(self, sources, *, future_s=0.0, period_s=0.0):
        self.sources = list(sources)
        self.signals = []
        self.images = {}
        self.samples = []
        for ep, src in enumerate(self.sources):
            with np.load(src.path, allow_pickle=False) as z:
                signals = {
                    key: z[key]
                    for key in (
                        "observation_ns",
                        "state_ns",
                        "state_log_ns",
                        "action_ns",
                        "state",
                        "action",
                        "commands",
                        "command_ns",
                        "camera_header_ns",
                    )
                }
            rows, targets = sample_indices(
                signals["observation_ns"],
                signals["state_ns"],
                signals["action_ns"],
                signals["command_ns"],
                round(future_s * 1e9),
                round(period_s * 1e9),
            )
            self.signals.append(signals)
            self.samples.extend((ep, int(i), int(j)) for i, j in zip(rows, targets, strict=True))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        ep, row, target = self.samples[index]
        source = self.sources[ep]
        if ep not in self.images:
            self.images[ep] = [read_npz_member(source.path, key + ".npy") for key in CAMERAS]
        s = self.signals[ep]
        return dict(
            raw=np.stack([a[row] for a in self.images[ep]]),
            state=s["state"][row],
            action=s["action"][row],
            future_action=s["commands"][target],
            observation_ns=s["observation_ns"][row],
            state_ns=s["state_ns"][row],
            state_log_ns=s["state_log_ns"][row],
            action_ns=s["action_ns"][row],
            future_action_ns=s["command_ns"][target],
            camera_header_ns=s["camera_header_ns"][row],
            episode=source.group_id,
            subset=source.suite,
        )
