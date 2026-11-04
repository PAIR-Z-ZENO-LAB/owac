"""Two-GPU native multi-task evaluation of a compact checkpoint."""

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from owac.runtime.compact_swanlab import publish_results
from owac.runtime.j5_action_finish import simulation_environment, wait_for_server


def evaluation_shards(cfg):
    """Partition task IDs while retaining the original suite's RNG and GPU mapping."""
    closed_loop = cfg["closed_loop"]
    suites = closed_loop["suites"]
    task_shards = closed_loop.get("task_shards_per_suite", 1)
    for task_shard in range(task_shards):
        task_ids = closed_loop["task_ids"][task_shard::task_shards]
        if not task_ids:
            continue
        for suite_index, suite in enumerate(suites):
            local = copy.deepcopy(cfg)
            local["closed_loop"].update(
                suites=[suite],
                task_ids=task_ids,
                suite_seed_indices={name: index for index, name in enumerate(suites)},
            )
            yield suite_index % 2, local


def wait_for_rollouts(clients, servers, output):
    while any(process.poll() is None for process in clients):
        for process in clients + servers:
            if process.poll() not in (None, 0):
                raise RuntimeError(
                    f"Native evaluation process exited {process.returncode}; see {output}"
                )
        time.sleep(5)
    # The last process may fail between polls, when no client remains running.
    for process in clients:
        code = process.wait()
        if code:
            raise RuntimeError(f"Native evaluation process exited {code}; see {output}")


def run(
    cfg,
    checkpoint,
    output,
    *,
    episodes=None,
    initial_offset=None,
    lighting=None,
    evaluation_kind=None,
):
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    step = int(payload["step"])
    del payload
    output.mkdir(parents=True, exist_ok=True)
    servers, clients, streams = [], [], []
    shards = []
    try:
        for gpu in range(2):
            port = 9036 + gpu
            stream = (output / f"server-{gpu}.log").open("w")
            streams.append(stream)
            openpi = Path(cfg["simulation"]["openpi_root"])
            server_env = dict(os.environ)
            server_env["PYTHONPATH"] = ":".join(
                [
                    str(Path(__file__).resolve().parents[2]),
                    str(openpi / "src"),
                    str(openpi / "packages/openpi-client/src"),
                ]
            )
            server = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "owac.runtime.compact_policy_server",
                    "--checkpoint",
                    str(checkpoint),
                    "--device",
                    f"cuda:{gpu}",
                    "--port",
                    str(port),
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=server_env,
            )
            servers.append(server)
            wait_for_server(server, port)
        # Policies have no episode state; each client supplies its own fixed noise.
        # Concurrent simulators can share a model without changing policy inputs.
        for shard_index, (gpu, local) in enumerate(evaluation_shards(cfg)):
            shard = output / f"shard-{shard_index}"
            shard.mkdir(exist_ok=True)
            if episodes is not None:
                local["closed_loop"]["seeds_per_task"] = episodes
            if initial_offset is not None:
                local["closed_loop"]["initial_offset"] = initial_offset
            if lighting is not None:
                local["closed_loop"]["lighting"] = lighting
            config = shard / "config.json"
            config.write_text(json.dumps(local, ensure_ascii=False, indent=2))
            port = 9036 + gpu
            env = simulation_environment(local, shard)
            # EGL and inference use the same physical GPU for this shard.
            env["MUJOCO_EGL_DEVICE_ID"] = str(gpu)
            stream = (shard / "rollouts.log").open("w")
            streams.append(stream)
            clients.append(
                subprocess.Popen(
                    [
                        cfg["simulation"]["python"],
                        "-m",
                        "owac.simulation.j5_action_eval",
                        "--config",
                        str(config),
                        "--output",
                        str(shard / "raw"),
                        "--author-root",
                        cfg["simulation"]["author_root"],
                        "--role",
                        "raw",
                        "--port",
                        str(port),
                    ],
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
            )
            shards.append(shard)
        wait_for_rollouts(clients, servers, output)
        rows = []
        for shard in shards:
            rows.extend(
                json.loads(line) for line in (shard / "raw/episodes.jsonl").read_text().splitlines()
            )
        groups = {}
        for row in rows:
            group = groups.setdefault(
                f"{row['suite']}/{row['lighting']}", {"successes": 0, "episodes": 0}
            )
            group["successes"] += int(row["success"])
            group["episodes"] += 1
        result = {
            "successes": sum(r["success"] for r in rows),
            "episodes": len(rows),
            "groups": groups,
            "checkpoint": str(checkpoint),
            "step": step,
            "evaluation_kind": evaluation_kind,
            "native_exit": 0,
        }
        (output / "summary.json").write_text(json.dumps(result, indent=2))
        (output / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        log_result(cfg, result, output)
        print(json.dumps(result), flush=True)
        return result
    finally:
        for process in clients + servers:
            if process.poll() is None:
                process.terminate()
                process.wait()
        for stream in streams:
            stream.close()


def log_result(cfg, result, output):
    publish_results(cfg, [(output, result)], output / "swanlab-training")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--initial-offset", type=int)
    parser.add_argument("--lighting", nargs="+")
    parser.add_argument(
        "--evaluation-kind", choices=["screening", "development", "independent-final"]
    )
    args = parser.parse_args()
    run(
        json.loads(args.config.read_text()),
        args.checkpoint,
        args.output,
        episodes=args.episodes,
        initial_offset=args.initial_offset,
        lighting=args.lighting,
        evaluation_kind=args.evaluation_kind,
    )


if __name__ == "__main__":
    main()
