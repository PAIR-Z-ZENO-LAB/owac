"""Reconstruct lossless observations along a recorded J5 action trajectory."""

import argparse
import json
from pathlib import Path

import numpy as np

from owac.simulation.rawvla_teacher_controller import (
    _load_author_replay_module,
    raw_observation_frames,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    author = _load_author_replay_module(Path(cfg["simulation"]["author_root"]))
    from libero.libero import benchmark
    from rawvla_bench.libero import apply_entry_lighting
    from rawvla_bench.lighting import rawvla_light_xml_from_string
    from rawvla_bench.raw import make_rawvla_observation

    args.output.mkdir(parents=True, exist_ok=True)
    records = [json.loads(x) for x in (args.episodes / "episodes.jsonl").read_text().splitlines()]
    receipts = []
    for light in ["Normal", "ExtremeLow"]:
        row = next(
            x
            for x in records
            if x["suite"] == "libero_spatial"
            and x["task"] == 0
            and x["initial_index"] == 0
            and x["lighting"] == light
        )
        directory = args.episodes / f"libero_spatial-00-{light}-00"
        initial = np.load(directory / "initial.npz")
        actions = np.load(directory / "actions.npz")
        suite = benchmark.get_benchmark_dict()[row["suite"]]()
        task = suite.get_task(row["task"])
        env = author._make_env(task, row["seed"])
        env.seed(row["seed"])
        env.reset()
        env.reset_from_xml_string(rawvla_light_xml_from_string(env.sim.model.get_xml()))
        env.sim.reset()
        apply_entry_lighting(env, row["lighting_parameters"])
        obs = env.set_init_state(initial["state"])
        for _ in range(10):
            obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
        for step, action in enumerate(actions["executed"]):
            if step in [0, 5, 20, 50, 100, 150, 200]:
                frames = list(raw_observation_frames(obs).values())
                raw = np.stack(
                    [
                        make_rawvla_observation(
                            frame,
                            noise_seed=row["seed"] ^ 0x5A17,
                            frame_index=step + 10,
                            view_index=view,
                            representation="raw",
                            sensor_saturation_ev=row["lighting_parameters"]["sensor_saturation_ev"],
                        )
                        for view, frame in enumerate(frames)
                    ]
                )
                if step == 0:
                    delta = np.abs(raw.astype(float) - initial["inputs"].astype(float))
                    receipts.append(
                        {
                            "lighting": light,
                            "initial_raw_max_abs_difference": float(delta.max()),
                            "initial_raw_mean_abs_difference": float(delta.mean()),
                        }
                    )
                np.savez(
                    args.output / f"{light}-{step:03d}.npz",
                    raw=raw,
                    display=np.stack(frames),
                    noise=actions["noises"][step // 5],
                    recorded_plan=actions["plans"][step // 5],
                    prompt=task.language,
                    step=step,
                )
            obs, _, _, _ = env.step(action)
        env.close()
    (args.output / "replay.json").write_text(json.dumps(receipts, indent=2))
    print(json.dumps(receipts), flush=True)


if __name__ == "__main__":
    main()
