"""Train a frozen bridge, compare selective expert LoRA, then evaluate the winner."""

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
    cfg = json.loads(args.config.read_text())

    def train(config, output, resume=None):
        (root / "pipeline-status.json").write_text(
            json.dumps({"phase": output.name, "status": "running", "time": time.time()})
        )
        output.mkdir(parents=True, exist_ok=True)
        config_path = output / "production.json"
        config_path.write_text(json.dumps(config, indent=2))
        command = [
            sys.executable,
            "-m",
            "owac.runtime.j5_action_launch",
            "--config",
            str(config_path),
            "--output",
            str(output),
            "--no-evaluation",
        ]
        if resume:
            command += ["--resume", str(resume)]
            # Keep the already measured best state when continuing in a new directory.
            source = resume.parent / "best.pt"
            if source.exists():
                os.link(source, output / "best.pt")
        subprocess.run(command, check=True)
        status = json.loads((output / "status.json").read_text())
        if status["status"] != "completed":
            (root / "pipeline-status.json").write_text(json.dumps({"phase": output.name, **status}))
        return status["status"] == "completed"

    bridge = root / "bridge"
    if not train(cfg, bridge, args.resume):
        return
    base = root / "bridge-best-policy.pt"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "owac.training.j5_action_export",
            "--checkpoint",
            str(bridge / "best.pt"),
            "--output",
            str(base),
        ],
        check=True,
    )
    expert_cfg = json.loads(json.dumps(cfg))
    expert_cfg.pop("selection", None)
    expert_cfg["adaptation"].update(fresh_official=False, base=str(base), ranks=[0, 8])
    expert_cfg["training"].update(
        learning_rate=1e-5, expert_learning_rate=5e-6, micro_batch=64, warmup=300
    )
    expert_cfg["swanlab"] = {
        "id": "owac-j5-selective-expert-20261031",
        "name": "J5-ACT｜冻结视觉选择性动作LoRA｜LIBERO五照明｜V/O/MLP-r8｜6k｜20261031",
    }
    expert_cfg["differences"] += [
        "Initialize from best bridge; freeze RAW/language/QK/action-time interfaces",
        "Bridge peak1e-5; expert V/O/MLP rank8 peak5e-6; new Adam and stage schedule",
    ]
    expert = root / "expert"
    if not train(expert_cfg, expert):
        return
    candidates = {
        name: json.loads((path / "selection.json").read_text())
        for name, path in [("bridge", bridge), ("expert", expert)]
    }
    winner = min(candidates, key=lambda name: candidates[name]["best"])
    (root / "comparison.json").write_text(
        json.dumps(
            {
                "selection": "fixed validation sampled MAE7, not closed-loop success",
                "candidates": candidates,
                "winner": winner,
            },
            indent=2,
        )
    )
    (root / "pipeline-status.json").write_text(
        json.dumps({"phase": "closed-loop", "winner": winner, "status": "running"})
    )
    selected = root / winner
    subprocess.run(
        [
            sys.executable,
            "-m",
            "owac.runtime.j5_action_finish",
            "--config",
            str(selected / "production.json"),
            "--output",
            str(selected),
        ],
        check=True,
    )

    (root / "pipeline-status.json").write_text(
        json.dumps(
            {"phase": "closed-loop", "winner": winner, "status": "completed", "time": time.time()}
        )
    )


if __name__ == "__main__":
    main()
