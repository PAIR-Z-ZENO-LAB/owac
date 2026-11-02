"""Reconstruct both checkpoint trajectories for the three paired regressions."""

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
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads((args.root / "production.json").read_text())
    author = _load_author_replay_module(Path(cfg["simulation"]["author_root"]))
    from libero.libero import benchmark
    from rawvla_bench.libero import apply_entry_lighting
    from rawvla_bench.lighting import rawvla_light_xml_from_string
    from rawvla_bench.raw import make_rawvla_observation

    args.output.mkdir(parents=True, exist_ok=True)
    receipts = []
    for checkpoint in [3000, 6000]:
        episodes = args.root / f"development-{checkpoint:06d}/closed-loop/raw"
        rows = [json.loads(x) for x in (episodes / "episodes.jsonl").read_text().splitlines()]
        for suite_name, initial_index in [
            ("libero_spatial", 7),
            ("libero_object", 6),
            ("libero_10", 7),
        ]:
            row = next(
                x for x in rows if x["suite"] == suite_name and x["initial_index"] == initial_index
            )
            name = f"{suite_name}-00-Normal-{initial_index:02d}"
            directory = args.output / str(checkpoint) / name
            directory.mkdir(parents=True, exist_ok=True)
            with np.load(episodes / name / "initial.npz") as z:
                state, initial_raw = z["state"], z["inputs"]
            with np.load(episodes / name / "actions.npz") as z:
                actions, plans, noises = z["executed"], z["plans"], z["noises"]
            task = benchmark.get_benchmark_dict()[suite_name]().get_task(0)
            env = author._make_env(task, row["seed"])
            env.seed(row["seed"])
            env.reset()
            env.reset_from_xml_string(rawvla_light_xml_from_string(env.sim.model.get_xml()))
            env.sim.reset()
            apply_entry_lighting(env, row["lighting_parameters"])
            obs = env.set_init_state(state)
            for _ in range(10):
                obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
            first_success, trajectory = None, []
            for step, action in enumerate(actions):
                trajectory.append(np.r_[obs["robot0_eef_pos"], obs["robot0_gripper_qpos"]])
                if step % 5 == 0:
                    frames = list(raw_observation_frames(obs).values())
                    images = {}
                    for key, representation in [("raw", "raw"), ("rgb", "default_isp")]:
                        images[key] = np.stack(
                            [
                                make_rawvla_observation(
                                    frame,
                                    noise_seed=row["seed"] ^ 0x5A17,
                                    frame_index=step + 10,
                                    view_index=view,
                                    representation=representation,
                                    sensor_saturation_ev=row["lighting_parameters"][
                                        "sensor_saturation_ev"
                                    ],
                                )
                                for view, frame in enumerate(frames)
                            ]
                        )
                    if step == 0:
                        initial_difference = int(
                            np.abs(images["raw"].astype(int) - initial_raw).max()
                        )
                    np.savez_compressed(
                        directory / f"{step:04d}.npz",
                        **images,
                        display=np.stack(frames),
                        noise=noises[step // 5],
                        recorded_plan=plans[step // 5],
                        prompt=task.language,
                        step=step,
                        eef=obs["robot0_eef_pos"],
                        gripper=obs["robot0_gripper_qpos"],
                        body_positions=env.sim.data.body_xpos.copy(),
                    )
                obs, _, _, _ = env.step(action)
                if env.check_success() and first_success is None:
                    first_success = step + 1
            receipt = dict(
                checkpoint=checkpoint,
                case=name,
                initial_raw_max_abs=initial_difference,
                recorded_success=row["success"],
                replay_success=bool(env.check_success()),
                first_success=first_success,
                executed_steps=len(actions),
                body_names=list(env.sim.model.body_names),
            )
            np.save(directory / "eef-trajectory.npy", np.asarray(trajectory))
            (directory / "receipt.json").write_text(json.dumps(receipt, indent=2))
            receipts.append(receipt)
            print(json.dumps({k: v for k, v in receipt.items() if k != "body_names"}), flush=True)
            env.close()
    (args.output / "receipts.json").write_text(json.dumps(receipts, indent=2))


if __name__ == "__main__":
    main()
