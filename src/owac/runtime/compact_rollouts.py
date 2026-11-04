"""Two-GPU native multi-task evaluation of a compact checkpoint."""

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from owac.runtime.j5_action_finish import simulation_environment, wait_for_server


def run(cfg, checkpoint, output, *, episodes=None, initial_offset=None, lighting=None):
    output.mkdir(parents=True, exist_ok=True)
    servers, clients, streams = [], [], []
    shards = []
    try:
        for gpu in range(2):
            shard = output / f"shard-{gpu}"
            shard.mkdir(exist_ok=True)
            local = copy.deepcopy(cfg)
            local["closed_loop"]["suite_seed_indices"] = {
                name: index for index, name in enumerate(cfg["closed_loop"]["suites"])
            }
            local["closed_loop"]["suites"] = cfg["closed_loop"]["suites"][gpu::2]
            if episodes is not None:
                local["closed_loop"]["seeds_per_task"] = episodes
            if initial_offset is not None:
                local["closed_loop"]["initial_offset"] = initial_offset
            if lighting is not None:
                local["closed_loop"]["lighting"] = lighting
            config = shard / "config.json"
            config.write_text(json.dumps(local, ensure_ascii=False, indent=2))
            port = 9036 + gpu
            stream = (shard / "server.log").open("w")
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
        while any(p.poll() is None for p in clients):
            for process in clients + servers:
                if process.poll() not in (None, 0):
                    raise RuntimeError(
                        f"Native evaluation process exited {process.returncode}; see {output}"
                    )
            time.sleep(5)
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
    import swanlab

    swanlab.login(api_key=Path("/root/.config/owac/swanlab-api-key").read_text().strip())
    swanlab.merge_settings(swanlab.Settings(probe=swanlab.Settings.Probe(monitor=False)))
    swanlab.init(
        project="OWAC",
        workspace="Jeong",
        id=f"{cfg['swanlab']['id']}-{output.name}",
        experiment_name=f"C1｜多任务闭环｜LIBERO RAW｜本体+任务语言｜{output.name}｜20261104",
        group=cfg["swanlab"]["group"],
        job_type="closed-loop",
        config=cfg,
        logdir=str(output / "swanlab"),
        resume="allow",
    )
    swanlab.define_metric("success/*", section_name="05 闭环评估")
    metrics = {
        f"success/{key}": value["successes"] / value["episodes"]
        for key, value in result["groups"].items()
    }
    metrics.update(
        {
            "success/overall": result["successes"] / result["episodes"],
            "success/episodes": result["episodes"],
        }
    )
    swanlab.log(metrics, step=result["episodes"])
    swanlab.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--initial-offset", type=int)
    parser.add_argument("--lighting", nargs="+")
    args = parser.parse_args()
    run(
        json.loads(args.config.read_text()),
        args.checkpoint,
        args.output,
        episodes=args.episodes,
        initial_offset=args.initial_offset,
        lighting=args.lighting,
    )


if __name__ == "__main__":
    main()
