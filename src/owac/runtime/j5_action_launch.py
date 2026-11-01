"""Run the two-GPU J5 action job and retain its native exit receipt."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--initial-eval", action="store_true")
    parser.add_argument("--no-evaluation", action="store_true")
    parser.add_argument("--until-step", type=int)
    parser.add_argument("--reset-patience", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        "-m",
        "owac.training.j5_action_train",
        "--config",
        str(args.config.resolve()),
        "--output",
        str(root),
    ]
    if args.resume:
        command.extend(["--resume", str(args.resume.resolve())])
    if args.steps:
        command.extend(["--steps", str(args.steps)])
    if args.until_step:
        command.extend(["--until-step", str(args.until_step)])
    if args.reset_patience:
        command.append("--reset-patience")
    if args.initial_eval:
        command.append("--initial-eval")
    env = dict(
        os.environ,
        OMP_NUM_THREADS="8",
        JAX_PLATFORMS="cpu",
        MPLBACKEND="Agg",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    )
    with (root / "train.log").open("a") as stream:
        child = subprocess.Popen(
            command, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        (root / "launch.json").write_text(
            json.dumps({"pid": child.pid, "command": command, "started_at": time.time()}, indent=2)
        )
        code = child.wait()
    (root / "train.exit.json").write_text(
        json.dumps({"native_exit": code, "ended_at": time.time()})
    )
    if args.until_step:
        (root / f"segment-{args.until_step:06d}.exit.json").write_text(
            json.dumps(
                {"native_exit": code, "ended_at": time.time(), "target_step": args.until_step}
            )
        )
    if code == 0 and not args.steps and not args.no_evaluation:
        status = json.loads((root / "status.json").read_text())
        if status["status"] == "completed":
            with (root / "evaluation.log").open("w") as stream:
                code = subprocess.call(
                    [
                        sys.executable,
                        "-m",
                        "owac.runtime.j5_action_finish",
                        "--config",
                        str(args.config.resolve()),
                        "--output",
                        str(root),
                    ],
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
            (root / "evaluation.exit.json").write_text(
                json.dumps({"native_exit": code, "ended_at": time.time()})
            )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
