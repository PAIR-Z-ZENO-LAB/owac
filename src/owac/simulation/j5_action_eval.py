"""LIBERO paired-condition rollouts; runs in the existing Python 3.11 simulator."""

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
from owac.training.j5_proprio import joint_gripper_from_observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--author-root", type=Path, required=True)
    parser.add_argument("--role", choices=["raw", "rgb"], required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--variant", help="Named module swap served by the diagnostic policy")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())["closed_loop"]
    author = _load_author_replay_module(args.author_root)
    from libero.libero import benchmark
    from rawvla_bench.libero import apply_entry_lighting
    from rawvla_bench.lighting import (
        DEFAULT_EV_RANGES,
        lighting_parameters_for_sample,
        rawvla_light_xml_from_string,
    )
    from rawvla_bench.raw import make_rawvla_observation

    client = WebsocketClientPolicy(host="127.0.0.1", port=args.port)
    args.output.mkdir(exist_ok=True, parents=True)
    suites = ["libero_goal"] if args.smoke else cfg["suites"]
    task_ids = [6] if args.smoke else cfg["task_ids"]
    lights = ["Normal"] if args.smoke else cfg["lighting"]
    episodes = 1 if args.smoke else cfg["seeds_per_task"]
    results = []
    horizons = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
    for suite_index, suite_name in enumerate(suites):
        suite = benchmark.get_benchmark_dict()[suite_name]()
        for task_id in task_ids:
            task = suite.get_task(task_id)
            states = suite.get_task_init_states(task_id)
            for light in lights:
                for initial_index in range(
                    cfg.get("initial_offset", 0), cfg.get("initial_offset", 0) + episodes
                ):
                    seed_suite = cfg.get("suite_seed_indices", {}).get(suite_name, suite_index)
                    seed = 20261030 + seed_suite * 1000 + task_id * 10 + initial_index
                    rng = np.random.default_rng(seed)
                    ev = float(rng.uniform(*DEFAULT_EV_RANGES[light]))
                    lighting = dict(
                        lighting_parameters_for_sample(light, ev, sample_unit=float(rng.random()))
                    )
                    name = f"{suite_name}-{task_id:02d}-{light}-{initial_index:02d}"
                    directory = args.output / name
                    directory.mkdir(exist_ok=True)
                    env = author._make_env(task, seed)
                    env.seed(seed)
                    env.reset()
                    env.reset_from_xml_string(rawvla_light_xml_from_string(env.sim.model.get_xml()))
                    env.sim.reset()
                    apply_entry_lighting(env, lighting)
                    obs = env.set_init_state(states[initial_index])
                    for _ in range(10):
                        obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
                    client.reset()
                    executed, plans, noises, proprioceptions = [], [], [], []
                    success = False
                    horizon = 5 if args.smoke else horizons[suite_name]
                    with imageio.get_writer(directory / "rollout.mp4", fps=20) as video:
                        for step in range(horizon):
                            frames = raw_observation_frames(obs)
                            if step % cfg["execute_horizon"] == 0:
                                inputs = []
                                for view, frame in enumerate(frames.values()):
                                    inputs.append(
                                        make_rawvla_observation(
                                            frame,
                                            noise_seed=seed ^ 0x5A17,
                                            frame_index=step + 10,
                                            view_index=view,
                                            representation="raw"
                                            if args.role == "raw"
                                            else "default_isp",
                                            sensor_saturation_ev=lighting["sensor_saturation_ev"],
                                        )
                                    )
                                noise = (
                                    np.random.default_rng(seed * 100003 + step)
                                    .standard_normal(
                                        (
                                            cfg.get("action_horizon", 10),
                                            cfg.get("noise_action_dim", 32),
                                        )
                                    )
                                    .astype(np.float32)
                                )
                                prediction = client.infer(
                                    {
                                        **({"variant": args.variant} if args.variant else {}),
                                        args.role: np.stack(inputs),
                                        "prompt": task.language,
                                        "proprio": joint_gripper_from_observation(obs),
                                        "noise": noise,
                                    }
                                )
                                plan = prediction["actions"]
                                plans.append(plan)
                                noises.append(noise)
                                proprioceptions.append(joint_gripper_from_observation(obs))
                                if step == 0:
                                    np.savez(
                                        directory / "initial.npz",
                                        inputs=np.stack(inputs),
                                        proprio=joint_gripper_from_observation(obs),
                                        state=states[initial_index],
                                        noise=noise,
                                    )
                            video.append_data(np.concatenate(list(frames.values()), axis=1))
                            action = plan[step % cfg["execute_horizon"]]
                            obs, _, _, _ = env.step(action)
                            executed.append(action)
                            success = bool(env.check_success())
                            if success:
                                break
                    env.close()
                    np.savez(
                        directory / "actions.npz",
                        executed=np.asarray(executed),
                        plans=np.asarray(plans),
                        noises=np.asarray(noises),
                        proprio=np.asarray(proprioceptions),
                    )
                    record = {
                        "suite": suite_name,
                        "task": task_id,
                        "name": task.name,
                        "lighting": light,
                        "initial_index": initial_index,
                        "seed": seed,
                        "role": args.role,
                        "success": success,
                        "steps": len(executed),
                        "lighting_parameters": lighting,
                        "smoke_only": args.smoke,
                    }
                    results.append(record)
                    with (args.output / "episodes.jsonl").open("a") as stream:
                        stream.write(json.dumps(record) + "\n")
                    print(json.dumps(record), flush=True)
    summary = {}
    for suite_name in suites:
        for light in lights:
            rows = [r for r in results if r["suite"] == suite_name and r["lighting"] == light]
            summary[f"{suite_name}/{light}"] = {
                "successes": sum(r["success"] for r in rows),
                "episodes": len(rows),
            }
    (args.output / "summary.json").write_text(
        json.dumps({"role": args.role, "smoke_only": args.smoke, "groups": summary}, indent=2)
    )


if __name__ == "__main__":
    main()
