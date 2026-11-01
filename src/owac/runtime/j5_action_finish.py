"""Export the completed EMA policy and run paired LIBERO evaluation on both GPUs."""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def simulation_environment(cfg, output):
    legacy = Path(cfg["simulation"]["legacy_root"])
    runtime = legacy / "runtime"
    author = Path(cfg["simulation"]["author_root"])
    paths = [
        Path(__file__).resolve().parents[2],
        runtime / "sim",
        runtime / "audit",
        runtime / "policy",
        runtime / "libero-source",
        author,
        author / "experiments/simulation/libero",
        Path(cfg["simulation"]["openpi_root"]) / "packages/openpi-client/src",
    ]
    env = dict(
        os.environ,
        PYTHONPATH=":".join(map(str, paths)),
        MUJOCO_GL="egl",
        PYOPENGL_PLATFORM="egl",
        LIBERO_CONFIG_PATH=str(legacy / "libero-config"),
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        IMAGEIO_FFMPEG_EXE="/usr/bin/ffmpeg",
        PYTHONUNBUFFERED="1",
        NUMBA_CACHE_DIR=str(output / "numba-cache"),
        MPLCONFIGDIR=str(output / "mpl-cache"),
    )
    env["LD_LIBRARY_PATH"] = str(legacy / "runtime-gl") + ":" + env.get("LD_LIBRARY_PATH", "")
    env["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(legacy / "runtime-gl/nvidia.json")
    return env


def wait_for_server(process, port):
    while process.poll() is None:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1):
                return
        except urllib.error.URLError:
            time.sleep(1)
    raise RuntimeError(f"Policy server exited with {process.returncode}; see its log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    root = args.output.resolve()
    policy = root / "policy-final-ema.pt"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "owac.training.j5_action_export",
            "--checkpoint",
            str(
                root
                / (
                    ("best.pt" if cfg["training"].get("retain_best") else "latest.pt")
                    if cfg.get("adaptation", {}).get("kind") == "lora"
                    else "latest"
                )
            ),
            "--output",
            str(policy),
        ],
        check=True,
    )
    evaluation = root / "closed-loop"
    evaluation.mkdir(exist_ok=True)
    env = simulation_environment(cfg, evaluation)
    servers, clients, streams = [], [], []
    import swanlab

    swanlab.login(api_key=Path("/root/.config/owac/swanlab-api-key").read_text().strip())
    swanlab.init(
        project="OWAC",
        workspace="Jeong",
        id=cfg["closed_loop"]["swanlab_id"],
        resume="allow",
        experiment_name=f"{cfg['swanlab']['name']}｜闭环与官方RGB对照",
        config=cfg,
        logdir=str(evaluation / "swanlab"),
        group="J5-action",
        job_type="closed-loop",
    )
    try:
        for role, gpu, port in [("raw", 0, 9015), ("rgb", 1, 9016)]:
            checkpoint = policy if role == "raw" else Path(cfg["simulation"]["rgb_checkpoint"])
            command = [
                sys.executable,
                "-m",
                "owac.runtime.j5_policy_server",
                "--config",
                str(args.config.resolve()),
                "--checkpoint",
                str(checkpoint),
                "--device",
                f"cuda:{gpu}",
                "--port",
                str(port),
            ]
            if role == "rgb":
                command.append("--baseline")
            stream = (evaluation / f"{role}-server.log").open("w")
            streams.append(stream)
            server = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            servers.append(server)
        for (role, port), server in zip([("raw", 9015), ("rgb", 9016)], servers, strict=True):
            wait_for_server(server, port)
            command = [
                cfg["simulation"]["python"],
                "-m",
                "owac.simulation.j5_action_eval",
                "--config",
                str(args.config.resolve()),
                "--output",
                str(evaluation / role),
                "--author-root",
                cfg["simulation"]["author_root"],
                "--role",
                role,
                "--port",
                str(port),
            ]
            stream = (evaluation / f"{role}-rollouts.log").open("w")
            streams.append(stream)
            clients.append(
                subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
            )
        previous = -1
        while any(p.poll() is None for p in clients):
            if any(p.poll() not in (None, 0) for p in clients):
                raise RuntimeError(
                    "Closed-loop process failed; raw logs and completed episodes retained"
                )
            values, total = {}, 0
            for role in ["raw", "rgb"]:
                file = evaluation / role / "episodes.jsonl"
                rows = (
                    [json.loads(line) for line in file.read_text().splitlines()]
                    if file.exists()
                    else []
                )
                total += len(rows)
                values[f"progress/{role}/episodes"] = len(rows)
                values[f"progress/{role}/successes"] = sum(r["success"] for r in rows)
            if total != previous:
                swanlab.log(values, step=total)
                previous = total
            time.sleep(10)
        for process in clients:
            if process.returncode != 0:
                raise RuntimeError(f"Closed-loop exit {process.returncode}")
        summaries = {
            role: json.loads((evaluation / role / "summary.json").read_text())
            for role in ["raw", "rgb"]
        }
        (evaluation / "comparison.json").write_text(json.dumps(summaries, indent=2))
        metrics = {
            f"success/{role}/{key}": value["successes"] / value["episodes"]
            for role, result in summaries.items()
            for key, value in result["groups"].items()
        }
        swanlab.log(
            metrics,
            step=sum(sum(v["episodes"] for v in s["groups"].values()) for s in summaries.values()),
        )
        (evaluation / "exit.json").write_text(
            json.dumps({"native_exit": 0, "ended_at": time.time()})
        )
    finally:
        for process in clients + servers:
            if process.poll() is None:
                process.terminate()
                process.wait()
        for stream in streams:
            stream.close()
        swanlab.finish()


if __name__ == "__main__":
    main()
