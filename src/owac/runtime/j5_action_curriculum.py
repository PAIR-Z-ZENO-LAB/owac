"""Normal-light action adaptation with native development rollouts before five lights."""

import argparse
import json
import os
import subprocess
import sys
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
    normal = root / "normal"
    normal.mkdir(exist_ok=True)
    config_path = normal / "production.json"
    config_path.write_text(json.dumps(cfg, indent=2))
    resume = args.resume
    if resume and resume.parent != normal and not (normal / "best.pt").exists():
        os.link(resume.parent / "best.pt", normal / "best.pt")
    results = []

    def status(phase, **fields):
        (root / "curriculum-status.json").write_text(
            json.dumps({"phase": phase, **fields}, indent=2)
        )

    def train(config, output, restore=None, until=None, reset_patience=False):
        command = [
            sys.executable,
            "-m",
            "owac.runtime.j5_action_launch",
            "--config",
            str(config),
            "--output",
            str(output),
            "--no-evaluation",
        ]
        if restore:
            command += ["--resume", str(restore)]
        if cfg.get("validation", {}).get("action_metrics"):
            command.append("--initial-eval")
        if reset_patience:
            command.append("--reset-patience")
        if until:
            command += ["--until-step", str(until)]
        subprocess.run(command, check=True)
        return json.loads((output / "status.json").read_text())["status"] in (
            "completed",
            "segment_completed",
        )

    for step in cfg["development"]["checkpoints"]:
        status("normal-training", target_step=step)
        if not train(
            config_path,
            normal,
            resume,
            step,
            reset_patience=resume is not None and resume.parent != normal,
        ):
            status("paused", stage="normal")
            return
        checkpoint = normal / f"step-{step:06d}.pt"
        resume = normal / "latest.pt"
        development = root / f"development-{step:06d}"
        development.mkdir(exist_ok=True)
        dev_cfg = json.loads(json.dumps(cfg))
        dev_cfg["closed_loop"].update(cfg["development"])
        dev_cfg["closed_loop"]["swanlab_id"] = f"owac-j5-act2-dev-{step}-20261101"
        dev_cfg["closed_loop"]["swanlab_name"] = (
            f"J5-ACT2｜开发闭环｜LIBERO正常光｜RAW开发集｜第{step}步｜20261101"
        )
        dev_config = development / "config.json"
        dev_config.write_text(json.dumps(dev_cfg, indent=2))
        status("development-rollouts", step=step)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "owac.runtime.j5_action_finish",
                "--config",
                str(dev_config),
                "--output",
                str(development),
                "--checkpoint",
                str(checkpoint),
            ],
            check=True,
        )
        summary = json.loads((development / "closed-loop/raw/summary.json").read_text())
        rows = list(summary["groups"].values())
        record = {
            "step": step,
            "successes": sum(r["successes"] for r in rows),
            "episodes": sum(r["episodes"] for r in rows),
        }
        results.append(record)
        (root / "development-results.json").write_text(json.dumps(results, indent=2))
        # Compact checkpoint retains recovery; full exported dev policy is transient.
        (development / "policy-final-ema.pt").unlink()
    selected = max(results, key=lambda r: r["successes"])
    if selected["successes"] == 0:
        status(
            "normal-control-not-established",
            development=results,
            next="Diagnose complete failed development rollouts before expanding illumination",
        )
        return
    status("five-light-initialization", selected_normal_step=selected["step"])
    base = root / "normal-selected-policy.pt"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "owac.training.j5_action_export",
            "--checkpoint",
            str(normal / f"step-{selected['step']:06d}.pt"),
            "--output",
            str(base),
        ],
        check=True,
    )
    five = root / "five-light"
    five.mkdir(exist_ok=True)
    five_cfg = json.loads(json.dumps(cfg))
    five_cfg.get("monitoring", {}).pop("history", None)
    five_cfg["data"].pop("lighting")
    five_cfg["data"]["sampling"] = "uniform training frames; all five lighting conditions"
    five_cfg["adaptation"]["base"] = str(base)
    five_cfg["training"].update(
        learning_rate=1e-5,
        group_learning_rates={
            "bridge": 1e-5,
            "language": 2e-6,
            "expert_content": 2e-6,
            "expert_qk": 5e-7,
        },
    )
    five_cfg["monitoring"].pop("history", None)
    five_cfg["swanlab"] = {
        "id": "owac-j5-act2-five-mse-20261101",
        "project": cfg["swanlab"].get("project", "OWAC"),
        "group": cfg["swanlab"].get("group", "J5-action"),
        "name": "J5-ACT2｜RAW后端适配｜LIBERO五照明｜动作MSE监测｜正常光择优续接｜20261101",
    }
    five_config = five / "production.json"
    five_config.write_text(json.dumps(five_cfg, indent=2))
    status("five-light-training", selected_normal_step=selected["step"])
    if not train(five_config, five):
        status("paused", stage="five-light")
        return
    status("final-closed-loop")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "owac.runtime.j5_action_finish",
            "--config",
            str(five_config),
            "--output",
            str(five),
        ],
        check=True,
    )
    status("completed")


if __name__ == "__main__":
    main()
