"""Exercise the exported policy's live-history entrypoint on saved paired sensor records."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from owac.frameworks.raw_compact.factory import make_policy
from owac.runtime.teleop_history import causal_history_indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("history", "predict"))
    parser.add_argument("source", type=Path, help="Aligned directory or exported checkpoint")
    parser.add_argument("frames", type=Path)
    parser.add_argument("histories", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(3)
    with np.load(args.frames) as data:
        episodes, indices = data["episode"], data["index"]
        qs, raws = data["state"], data["raw"]
    if args.mode == "history":
        times, states = [], []
        for name, ix, state in zip(episodes, indices, qs, strict=True):
            with np.load(args.source / (name + ".npz")) as z:
                q, t = z["state"], z["observation_ns"]
            np.testing.assert_array_equal(state, q[ix])
            past = causal_history_indices(t, [24, 12, 6, 3, 1, 0], 1e9 / 30)[ix]
            states.append(q[past])
            times.append((t[past] - t[ix]) / 1e9)
        np.savez(args.histories, states=states, times=times)
        return
    payload = torch.load(args.source, weights_only=False, mmap=True, map_location="cpu")
    model = make_policy(payload["config"]).eval()
    model.load_state_dict(payload["model"], strict=True)
    with np.load(args.histories) as bank:
        states, times = bank["states"], bank["times"]
    predictions = []
    for raw, q, clock in zip(raws, states, times, strict=True):
        frames = [SimpleNamespace(data=r, width=640, height=480, stride=0) for r in raw]
        history = [
            SimpleNamespace(at=t, feedback=SimpleNamespace(values=v), frames=frames)
            for t, v in zip(clock, q, strict=True)
        ]
        predictions.append(model.predict_live(history, lambda data, *args: data))
    predictions = np.stack(predictions)
    if not np.isfinite(predictions).all() or predictions.shape != (len(qs), 12, 8):
        raise ValueError("Invalid exported action tensor")
    np.save(args.output, predictions)
    print(f"step={payload['step']} episodes={len(set(episodes))} pairs={len(qs)}")


if __name__ == "__main__":
    main()
