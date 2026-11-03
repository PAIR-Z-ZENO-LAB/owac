"""Run full native LIBERO episodes for explicit checkpoint module swaps."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from owac.runtime.j5_action_finish import simulation_environment, wait_for_server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--port", type=int, default=9026)
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / "weight-audit"
    cfg = json.loads((root / "production.json").read_text())
    cfg["closed_loop"].update(cfg["development"])
    config = out / "native-config.json"
    config.write_text(json.dumps(cfg, indent=2))
    env = simulation_environment(cfg, out)
    import swanlab

    swanlab.login(api_key=Path("/root/.config/owac/swanlab-api-key").read_text().strip())
    swanlab.merge_settings(swanlab.Settings(probe=swanlab.Settings.Probe(monitor=False)))
    label = "-".join(args.variants)
    swanlab.init(
        project="OWAC",
        workspace="Jeong",
        id=f"owac-j5-swap-{label}-20261103",
        resume="allow",
        experiment_name=f"J5-权重诊断｜原生闭环｜LIBERO正常光｜{label}｜3000与4000交换｜20261103",
        config=cfg,
        group="05 闭环评估",
        job_type="closed-loop",
        logdir=str(out / f"swanlab-{label}"),
    )
    swanlab.define_metric("progress/*", section_name="05 闭环评估")
    swanlab.define_metric("success/*", section_name="05 闭环评估")
    command = [
        sys.executable,
        "-m",
        "owac.runtime.j5_weight_policy",
        "--before",
        str(root / "normal/initial-003000.pt"),
        "--after",
        str(root / "normal/step-004000.pt"),
        "--device",
        args.device,
        "--port",
        str(args.port),
    ]
    child = None
    with (out / f"server-{args.port}.log").open("w") as stream:
        server = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        try:
            wait_for_server(server, args.port)
            for variant in args.variants:
                dest = out / "native" / variant
                dest.mkdir(parents=True, exist_ok=True)
                command = [
                    cfg["simulation"]["python"],
                    "-m",
                    "owac.simulation.j5_action_eval",
                    "--config",
                    str(config),
                    "--output",
                    str(dest),
                    "--author-root",
                    cfg["simulation"]["author_root"],
                    "--role",
                    "raw",
                    "--port",
                    str(args.port),
                    "--variant",
                    variant,
                ]
                with (dest / "rollouts.log").open("w") as log:
                    child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                    previous = -1
                    while child.poll() is None:
                        path = dest / "episodes.jsonl"
                        rows = (
                            [json.loads(x) for x in path.read_text().splitlines()]
                            if path.exists()
                            else []
                        )
                        if len(rows) != previous:
                            swanlab.log(
                                {
                                    f"progress/{variant}/episodes": len(rows),
                                    f"progress/{variant}/successes": sum(
                                        x["success"] for x in rows
                                    ),
                                },
                                step=len(rows),
                            )
                            previous = len(rows)
                        time.sleep(10)
                (dest / "exit.json").write_text(json.dumps(dict(native_exit=child.returncode)))
                if child.returncode:
                    raise RuntimeError(f"{variant} native exit {child.returncode}")
                summary = json.loads((dest / "summary.json").read_text())
                groups = list(summary["groups"].values())
                score = sum(x["successes"] for x in groups) / sum(x["episodes"] for x in groups)
                swanlab.log({f"success/{variant}": score}, step=20)
                print(variant, json.dumps(summary), flush=True)
        finally:
            for process in [child, server]:
                if process is not None and process.poll() is None:
                    process.terminate()
                    process.wait()
            swanlab.finish()


if __name__ == "__main__":
    main()
