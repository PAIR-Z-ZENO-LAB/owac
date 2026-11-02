"""Paired native closed loops with gripper and successful-approach interventions."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy

from owac.simulation.rawvla_teacher_controller import (
    _load_author_replay_module,
    raw_observation_frames,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=9025)
    args = parser.parse_args()
    cfg = json.loads((args.root / "production.json").read_text())
    author = _load_author_replay_module(Path(cfg["simulation"]["author_root"]))
    from libero.libero import benchmark
    from rawvla_bench.libero import apply_entry_lighting
    from rawvla_bench.lighting import rawvla_light_xml_from_string
    from rawvla_bench.raw import make_rawvla_observation

    source = args.root / "development-006000/closed-loop/raw"
    row = next(
        json.loads(x)
        for x in (source / "episodes.jsonl").read_text().splitlines()
        if json.loads(x)["suite"] == "libero_object" and json.loads(x)["initial_index"] == 6
    )
    case = source / "libero_object-00-Normal-06"
    with np.load(case / "initial.npz") as z:
        initial_state = z["state"]
    with np.load(case / "actions.npz") as z:
        reference_actions = z["executed"]
    with np.load(
        args.root / "development-003000/closed-loop/raw/libero_object-00-Normal-06/actions.npz"
    ) as z:
        successful_actions = z["executed"]
    client = WebsocketClientPolicy(host="127.0.0.1", port=args.port)
    task = benchmark.get_benchmark_dict()["libero_object"]().get_task(0)
    summaries = []
    for mode in ["baseline", "hold50-53", "prefix3000-until55"]:
        out = args.root / "regression-audit/contact" / mode
        out.mkdir(parents=True, exist_ok=True)
        env = author._make_env(task, row["seed"])
        env.seed(row["seed"])
        env.reset()
        env.reset_from_xml_string(rawvla_light_xml_from_string(env.sim.model.get_xml()))
        env.sim.reset()
        apply_entry_lighting(env, row["lighting_parameters"])
        obs = env.set_init_state(initial_state)
        for _ in range(10):
            obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
        executed, predicted = [], []
        with imageio.get_writer(out / "rollout.mp4", fps=20) as video:
            for step in range(280):
                frames = list(raw_observation_frames(obs).values())
                if step % 5 == 0:
                    raw = np.stack(
                        [
                            make_rawvla_observation(
                                frame,
                                noise_seed=row["seed"] ^ 0x5A17,
                                frame_index=step + 10,
                                view_index=view,
                                representation="raw",
                                sensor_saturation_ev=row["lighting_parameters"][
                                    "sensor_saturation_ev"
                                ],
                            )
                            for view, frame in enumerate(frames)
                        ]
                    )
                    noise = (
                        np.random.default_rng(row["seed"] * 100003 + step)
                        .standard_normal((10, 32))
                        .astype(np.float32)
                    )
                    plan = client.infer({"raw": raw, "prompt": task.language, "noise": noise})[
                        "actions"
                    ]
                video.append_data(np.concatenate(frames, axis=1))
                action = plan[step % 5].copy()
                predicted.append(action.copy())
                if mode == "hold50-53" and 50 <= step < 54:
                    action[6] = 1.0
                if mode == "prefix3000-until55" and step < 55:
                    action = successful_actions[step].copy()
                obs, _, _, _ = env.step(action)
                executed.append(action)
                success = bool(env.check_success())
                if success:
                    break
        env.close()
        executed = np.asarray(executed)
        np.savez(out / "actions.npz", executed=executed, predicted=np.asarray(predicted))
        receipt = dict(
            intervention=mode,
            success=success,
            steps=len(executed),
            pre_intervention_max_abs=float(np.abs(executed[:50] - reference_actions[:50]).max()),
        )
        if mode == "baseline":
            receipt["full_action_max_abs"] = float(
                np.abs(executed - reference_actions[: len(executed)]).max()
            )
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2))
        print(json.dumps(receipt), flush=True)
        summaries.append(receipt)
    (out.parent / "summary.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
