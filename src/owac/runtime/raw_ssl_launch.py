"""Launch the isolated two-GPU RAW SSL run and record its native exit status."""

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
        "owac.training.raw_ssl_train",
        "--config",
        str(args.config.resolve()),
        "--output",
        str(root),
    ]
    if args.resume:
        command.extend(["--resume", str(args.resume.resolve())])
    env = dict(os.environ, OMP_NUM_THREADS="8", MPLBACKEND="Agg")
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
    raise SystemExit(code)


if __name__ == "__main__":
    main()
