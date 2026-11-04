"""Train/evaluate successive checkpoints; preserve failure evidence and selection."""

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
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    training = root / "training"
    training.mkdir(exist_ok=True)
    results_file = root / "development.json"
    results = json.loads(results_file.read_text()) if results_file.exists() else []

    def execute(command, label):
        with (root / f"{label}.log").open("a") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        (root / f"{label}.exit.json").write_text(
            json.dumps(
                {"native_exit": result.returncode, "ended_at": time.time(), "command": command}
            )
        )
        if result.returncode:
            (root / "status.json").write_text(
                json.dumps({"phase": "failed", "stage": label, "native_exit": result.returncode})
            )
            raise RuntimeError(f"{label} exited {result.returncode}")

    for stage in cfg["research"]["stages"]:
        step = stage["step"]
        if any(r["step"] == step for r in results):
            continue
        status_file = training / "status.json"
        current = json.loads(status_file.read_text()).get("step", 0) if status_file.exists() else 0
        if current < step:
            (root / "status.json").write_text(
                json.dumps({"phase": "training", "target_step": step})
            )
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=2",
                "-m",
                "owac.training.compact_train",
                "--config",
                str(args.config.resolve()),
                "--output",
                str(training),
                "--stop-step",
                str(step),
            ]
            if (training / "latest.pt").exists():
                command += ["--resume", str(training / "latest.pt")]
            execute(command, f"train-{step:06d}")
        evaluation = root / f"development-{step:06d}"
        (root / "status.json").write_text(json.dumps({"phase": "development", "step": step}))
        if not (evaluation / "summary.json").exists():
            execute(
                [
                    sys.executable,
                    "-m",
                    "owac.runtime.compact_rollouts",
                    "--config",
                    str(args.config.resolve()),
                    "--checkpoint",
                    str(training / f"step-{step:06d}.pt"),
                    "--output",
                    str(evaluation),
                    "--episodes",
                    str(stage["episodes"]),
                    "--initial-offset",
                    "5",
                    "--lighting",
                    *stage["lighting"],
                ],
                f"eval-{step:06d}",
            )
        result = json.loads((evaluation / "summary.json").read_text()) | {"step": step}
        results.append(result)
        results_file.write_text(json.dumps(results, indent=2))
        # Final selection compares the identical full development protocol only.
        comparable = [r for r in results if r["episodes"] == cfg["research"]["selection_episodes"]]
        if comparable:
            selected = max(comparable, key=lambda r: r["successes"])
            (root / "selection.json").write_text(json.dumps(selected, indent=2))
        print(json.dumps({"review_checkpoint": result, "pid": os.getpid()}), flush=True)
    selected = json.loads((root / "selection.json").read_text())
    final = cfg["research"]["final"]
    evaluation = root / "independent-final"
    if not (evaluation / "summary.json").exists():
        (root / "status.json").write_text(
            json.dumps({"phase": "independent-final", "selected_step": selected["step"]})
        )
        execute(
            [
                sys.executable,
                "-m",
                "owac.runtime.compact_rollouts",
                "--config",
                str(args.config.resolve()),
                "--checkpoint",
                str(training / f"step-{selected['step']:06d}.pt"),
                "--output",
                str(evaluation),
                "--episodes",
                str(final["episodes"]),
                "--initial-offset",
                str(final["initial_offset"]),
                "--lighting",
                *final["lighting"],
            ],
            "independent-final",
        )
    (root / "status.json").write_text(
        json.dumps({"phase": "training-and-development-completed", "results": results})
    )


if __name__ == "__main__":
    main()
