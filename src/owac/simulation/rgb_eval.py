"""Paired lighting rollouts using current-light RGB and original policy controls."""

import argparse
import importlib.util
import json
import sys
import time
from collections import deque
from multiprocessing.connection import Client, Connection
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from owac.simulation.lighting_schedule import step_libero_observation
from owac.simulation.random_lighting import dynamic_result, entry_schedule, episode_identity
from owac.simulation.rawvla_teacher_controller import _load_author_replay_module
from owac.simulation.rgb_observation import (
    CAMERA_KEYS,
    CLEAN_PROTOCOL,
    RAW_PROTOCOL,
    SENSOR_PROTOCOL,
    policy_observation,
)

HORIZONS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
OBSERVATION_KEYS = (
    "agentview_image",
    "robot0_eye_in_hand_image",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)


def decode_actions(response, *, checkpoint_sha256=None):
    """Preserve the official client's numeric precision across Python environments."""
    if checkpoint_sha256 and response.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("Policy checkpoint does not match this evaluation")
    return np.asarray(response["actions"], dtype=response["dtype"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--entries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", required=True)
    transport = parser.add_mutually_exclusive_group(required=True)
    transport.add_argument("--port", type=int)
    transport.add_argument("--pipe-fd", type=int, help="inherited local policy connection")
    parser.add_argument("--engineering", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    protocol = cfg.get("observation_protocol", CLEAN_PROTOCOL)
    raw_policy = protocol == RAW_PROTOCOL
    author = _load_author_replay_module(cfg["author_root"])
    from libero.libero import benchmark
    from rawvla_bench.libero import apply_entry_lighting
    from rawvla_bench.lighting import rawvla_light_xml_from_string

    client = (
        Connection(args.pipe_fd)
        if args.pipe_fd is not None
        else Client(("127.0.0.1", args.port), authkey=b"owac-native-rgb-benchmark")
    )
    request_context = {}
    calls = 0

    def infer(data):
        nonlocal calls
        client.send({**request_context, **data})
        response = client.recv()
        if "error" in response:
            raise RuntimeError(response["error"])
        calls += 1
        return decode_actions(
            response, checkpoint_sha256=cfg["policies"][args.policy].get("checkpoint_sha256")
        )

    xvla = None
    if args.policy == "xvla":
        path = Path(cfg["policies"]["xvla"]["source"]) / "evaluation/libero/libero_client.py"
        spec = importlib.util.spec_from_file_location("native_xvla_client", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        class LocalClient(module.ClientModel):
            def _post(self, payload):
                return infer({"payload": payload})

        xvla = LocalClient("127.0.0.1", args.port)

    args.output.mkdir(parents=True, exist_ok=True)
    entries = json.loads(args.entries.read_text())
    result_path = args.output / "episodes.jsonl"
    done_keys = set()
    if result_path.exists():
        for line in result_path.read_text().splitlines():
            done_keys.add(json.loads(line)["id"])
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
                if xvla is not None:
                    xvla.reset()
                    for robot in env.env.robots:
                        robot.controller.use_delta = False
                horizon = HORIZONS[suite_name]
                if xvla is not None:
                    horizon = module.LIBERO_DATASETS_HORIZON[suite_name]
                plan, actions_log, calls = deque(), [], 0
                success = False
                if entry["base_seed_id"] == 0 or args.engineering:
                    video = imageio.get_writer(directory / "rollout.mp4", fps=20)
                for step in range(horizon):
                    calls_before = calls
                    request_context = {
                        "seed": int(entry["noise_seed"]) + step * 100003,
                        "suite": suite_name,
                        "prompt": task.language,
                    }
                    # X-VLA owns its action queue; supply each current observation
                    # without changing that queue or its official state lifecycle.
                    needs_image = xvla is not None or not plan or video is not None
                    policy_obs = (
                        policy_observation(obs, entry, frame_index=step + 10, protocol=protocol)
                        if needs_image
                        else obs
                    )
                    if step == 0 and (entry["base_seed_id"] == 0 or args.engineering):
                        np.savez_compressed(
                            directory / "first-observation.npz",
                            **{f"rendered_{key}": obs[key] for key in CAMERA_KEYS},
                            **{f"policy_{key}": policy_obs[key] for key in CAMERA_KEYS},
                        )
                    if video is not None:
                        video.append_data(
                            np.concatenate([policy_obs[k][::-1, ::-1] for k in CAMERA_KEYS], axis=1)
                        )
                    if xvla is not None:
                        controller = env.env.robots[0].controller
                        policy_obs["robo_pos"] = controller.ee_pos
                        policy_obs["robo_ori"] = xvla.processor.Mat_to_Rotate6D(
                            controller.ee_ori_mat
                        )
                        action = xvla.step(policy_obs, task.language)
                    else:
                        if not plan:
                            if raw_policy:
                                request = {
                                    "raw": np.stack(
                                        [policy_obs[k][::-1, ::-1] for k in CAMERA_KEYS]
                                    ),
                                    "proprio": np.concatenate(
                                        (obs["robot0_joint_pos"], obs["robot0_gripper_qpos"])
                                    ),
                                }
                            else:
                                request = {
                                    "observation": {k: policy_obs[k] for k in OBSERVATION_KEYS}
                                }
                            actions = infer(request)
                            execute = cfg["policies"][args.policy].get(
                                "execute_horizon", len(actions)
                            )
                            plan.extend(actions[:execute])
                        action = plan.popleft()
                    if calls > calls_before:
                        observation_steps.append(step)
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
                        raise RuntimeError(
                            "Native done without task success; inspect simulator verdict"
                        )
                np.save(directory / "actions.npy", np.asarray(actions_log))
                row = {
                    "id": identity,
                    "policy": args.policy,
                    "checkpoint_step": cfg["policies"][args.policy].get("checkpoint_step"),
                    "checkpoint_sha256": cfg["policies"][args.policy].get("checkpoint_sha256"),
                    "frontend": "raw-encoder"
                    if raw_policy
                    else ("fixed-isp" if protocol == SENSOR_PROTOCOL else "native-rgb"),
                    "observation_protocol": protocol,
                    "input_route": (
                        "current-light-render->noisy-RAW10->view8->RAW-encoder->action"
                        if raw_policy
                        else "current-light-render->noisy-RAW10->fixed-ISP->native-RGB-policy"
                        if protocol == SENSOR_PROTOCOL
                        else "current-light-rendered-RGB"
                    ),
                    "sensor_noise": cfg.get("sensor_noise", None),
                    "video_content": "RAW-view8"
                    if raw_policy
                    else "policy-input-RGB-before-native-camera-preprocessing",
                    "suite": suite_name,
                    "task": task_id,
                    "language": task.language,
                    "lighting": entry["lighting_domain"],
                    "initial_index": entry["init_state_index"],
                    "seed": entry["base_seed"],
                    "success": success,
                    "steps": len(actions_log),
                    "max_steps": horizon,
                    "common_budget_success": success and len(actions_log) <= HORIZONS[suite_name],
                    "action_mode": "absolute" if xvla is not None else "delta",
                    "action_dtype": str(np.asarray(actions_log).dtype),
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
                    json.dumps({k: row[k] for k in ("id", "success", "steps", "seconds")}),
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
