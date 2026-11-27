"""Execute one shard of the frozen, paired RawVLA-Bench LIBERO manifest."""

import argparse
import json
import time
from collections import deque
from multiprocessing.connection import Client
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from owac.simulation.lighting_schedule import step_libero_observation
from owac.simulation.random_lighting import dynamic_result, entry_schedule, episode_identity
from owac.simulation.rawvla_teacher_controller import (
    _load_author_replay_module,
    _state_from_obs,
    raw_observation_frames,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--entries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--frontend", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--engineering", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    author = _load_author_replay_module(cfg["author_root"])
    from libero.libero import benchmark
    from rawvla_bench.libero import apply_entry_lighting
    from rawvla_bench.lighting import rawvla_light_xml_from_string
    from rawvla_bench.raw import make_rawvla_observation

    args.output.mkdir(parents=True, exist_ok=True)
    client = Client(("127.0.0.1", args.port), authkey=b"owac-local-isp-benchmark")
    entries = json.loads(args.entries.read_text())
    done_keys = set()
    result_path = args.output / "episodes.jsonl"
    if result_path.exists():
        for line in result_path.read_text().splitlines():
            done_keys.add(json.loads(line)["id"])
    horizons = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
    stride = 12 if args.policy == "corp" else 5
    suites = {}
    with result_path.open("a") as results:
        for entry in entries:
            identity = episode_identity(entry)
            if identity in done_keys:
                continue
            started = time.monotonic()
            suite_name, task_id = entry["suite"], entry["task_id"]
            if suite_name not in suites:
                suites[suite_name] = benchmark.get_benchmark_dict()[suite_name]()
            suite = suites[suite_name]
            task = suite.get_task(task_id)
            initial = suite.get_task_init_states(task_id)[entry["init_state_index"]]
            env = author._make_env(task, int(entry["base_seed"]))
            directory = args.output / identity.replace("/", "-")
            if "dynamic_id" in entry and directory.exists():
                directory.rename(directory.with_name(directory.name + f"-attempt-{time.time_ns()}"))
            # One fixed initial state per task/light has video, all 10k retain actions.
            record_video = entry["base_seed_id"] == 0 or args.engineering
            directory.mkdir(exist_ok=True)
            schedule = entry_schedule(entry, directory)
            lighting = entry
            observation_steps = []
            video = None
            try:
                env.seed(int(entry["base_seed"]))
                env.reset()
                env.reset_from_xml_string(rawvla_light_xml_from_string(env.sim.model.get_xml()))
                env.sim.reset()
                if hasattr(env, "_rawvla_bench_lighting_baseline"):
                    delattr(env, "_rawvla_bench_lighting_baseline")
                apply_entry_lighting(env, entry)
                obs = env.set_init_state(initial)
                for _ in range(10):
                    obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
                client.send({"reset": True})
                client.recv()
                history = deque(maxlen=6 if args.frontend == "rawvla" else 1)
                actions_log, calls = [], 0
                success = False
                if record_video:
                    video = imageio.get_writer(directory / "rollout.mp4", fps=20)
                for step in range(horizons[suite_name]):
                    frames = list(raw_observation_frames(obs).values())
                    if args.frontend != "default" or step % stride == 0:
                        images = np.stack(
                            [
                                make_rawvla_observation(
                                    frame,
                                    noise_seed=entry["noise_seed"],
                                    frame_index=step + 10,
                                    view_index=view,
                                    representation="default_isp"
                                    if args.frontend == "default"
                                    else "raw",
                                    sensor_saturation_ev=entry["sensor_saturation_ev"],
                                )
                                for view, frame in enumerate(frames)
                            ]
                        )
                        history.append(images)
                    if step % stride == 0:
                        request = {
                            "state": _state_from_obs(obs),
                            "prompt": task.language,
                            "seed": int(entry["noise_seed"]) + step * 100003,
                        }
                        if args.frontend == "default":
                            request["rgb"] = images
                        else:
                            history_list = [history[0]] * (history.maxlen - len(history)) + list(
                                history
                            )
                            request["raw"] = np.stack(history_list, axis=1)
                        client.send(request)
                        response = client.recv()
                        if "error" in response:
                            raise RuntimeError(response["error"])
                        plan = np.asarray(response["actions"], np.float32)
                        calls += 1
                        observation_steps.append(step)
                    if video is not None:
                        video.append_data(np.concatenate(frames, axis=1))
                    action = plan[step % stride]
                    transition, lighting = step_libero_observation(
                        env,
                        action,
                        observation_step=step + 1,
                        schedule=schedule,
                        lighting=lighting,
                        seed=entry["base_seed"],
                    )
                    obs, _, native_done, _ = transition
                    actions_log.append(action)
                    success = bool(env.check_success())
                    if success:
                        break
                    if native_done:
                        raise RuntimeError("Native done without task success")
                np.save(directory / "actions.npy", np.asarray(actions_log, np.float32))
                row = {
                    "id": identity,
                    "policy": args.policy,
                    "frontend": args.frontend,
                    "suite": suite_name,
                    "task": task_id,
                    "language": task.language,
                    "lighting": entry["lighting_domain"],
                    "initial_index": entry["init_state_index"],
                    "seed": entry["base_seed"],
                    "success": success,
                    "steps": len(actions_log),
                    "policy_calls": calls,
                    "seconds": time.monotonic() - started,
                    "lighting_parameters": {
                        k: v
                        for k, v in entry.items()
                        if k not in ("lighting_schedule", "renderer_by_domain")
                    },
                    "formal": not args.engineering,
                    "native_exit": 0,
                    **dynamic_result(entry, schedule, observation_steps),
                }
                results.write(json.dumps(row) + "\n")
                results.flush()
                print(
                    json.dumps({key: row[key] for key in ("id", "success", "steps", "seconds")}),
                    flush=True,
                )
            finally:
                if video is not None:
                    video.close()
                env.close()
    client.close()
    (args.output / "complete.json").write_text(
        json.dumps({"native_exit": 0, "episodes": len(entries)})
    )


if __name__ == "__main__":
    main()
