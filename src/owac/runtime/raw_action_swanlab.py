"""Incrementally upload R1's existing action metrics without interrupting training."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def read_rows(path: Path, offset: int):
    if not path.exists():
        return [], offset
    rows = []
    with path.open("rb") as stream:
        stream.seek(offset)
        while line := stream.readline():
            if not line.endswith(b"\n"):
                break
            rows.append(json.loads(line))
            offset = stream.tell()
    return rows, offset


def metrics(row, validation=False):
    if validation:
        return {
            f"validation/{group}/{key}": value
            for group, values in row["groups"].items()
            for key, value in values.items()
        }
    return {
        "train/flow_loss": row["flow_loss"],
        "train/learning_rate": row["lr"],
        "train/update_seconds": row["seconds"],
        "train/presentations": row["presentations"],
        "train/epoch": row["epoch"],
        "train/optimizer_updates": row["step"],
    }


def resource_metrics(root, storage):
    result = {}
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for line in output.splitlines():
        index, utilization, used, total = (int(value.strip()) for value in line.split(","))
        result[f"gpu/{index}/utilization_percent"] = utilization
        result[f"gpu/{index}/memory_used_mb"] = used
        result[f"gpu/{index}/memory_total_mb"] = total
    run_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    result["budget/owned_gib"] = (storage["existing_owned_bytes"] + run_bytes) / 2**30
    result["budget/cap_gib"] = storage["cap_bytes"] / 2**30
    result.update(host_memory_metrics(root))
    return result


def host_memory_metrics(root, *, proc=Path("/proc"), cgroup=Path("/sys/fs/cgroup")):
    """Report reclaimable file cache separately from process/anonymous memory."""
    memory = dict(line.split() for line in (cgroup / "memory.stat").read_text().splitlines())
    events = dict(line.split() for line in (cgroup / "memory.events").read_text().splitlines())
    result = {
        "host/cgroup_memory_gib": int((cgroup / "memory.current").read_text()) / 2**30,
        "host/anonymous_memory_gib": int(memory["anon"]) / 2**30,
        "host/file_cache_gib": int(memory["file"]) / 2**30,
        "host/mapped_file_gib": int(memory["file_mapped"]) / 2**30,
        "host/oom_kills": int(events["oom_kill"]),
    }
    launch = root / "launch.json"
    if launch.exists():
        parent = str(json.loads(launch.read_text())["pid"])
        children = proc / parent / "task" / parent / "children"
        if children.exists():
            for rank, pid in enumerate(children.read_text().split()):
                try:
                    status = dict(
                        line.split(":", 1)
                        for line in (proc / pid / "status").read_text().splitlines()
                    )
                    result[f"host/worker{rank}/rss_gib"] = int(status["VmRSS"].split()[0]) / 2**20
                    result[f"host/worker{rank}/open_fds"] = len(list((proc / pid / "fd").iterdir()))
                except FileNotFoundError:
                    # Normal process-exit race; unavailable is not reported as zero.
                    continue
    return result


def main():
    import swanlab

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", default="owac-r1-action-100k-20261027")
    parser.add_argument("--key-file", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root
    state_path = root / "swanlab-r1-state.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {
            "updates_offset": 0,
            "validation_offset": 0,
            "train_step": 0,
            "validation_step": None,
        }
    )
    swanlab.login(api_key=args.key_file.read_text().strip(), save=False)
    config = json.loads((root / "config.json").read_text())
    run = swanlab.init(
        project="OWAC",
        name="R1｜动作监督｜LIBERO五照明｜全视觉头｜100k｜20261027",
        group="R-action-adaptation",
        job_type="flow-matching",
        tags=[
            "stage:R1",
            "objective:action-fm",
            "data:libero",
            "lighting:five",
            "scope:full-visual",
            "target:100k",
        ],
        id=args.run_id,
        resume="allow",
        mode="online",
        public=False,
        config=config,
        logdir=str(root / "swanlab"),
        settings=swanlab.Settings(
            probe={
                "hardware": False,
                "runtime": False,
                "requirements": False,
                "conda": False,
                "git": False,
                "swanlab": False,
                "monitor": False,
            },
            terminal={"proxy_type": "none"},
        ),
    )
    state["url"] = run.url
    last_resources_at = 0.0
    while True:
        exit_path = root / "train-r3.exit.json"
        finished = exit_path.exists()
        events = []
        for filename, offset_key, step_key, is_validation in (
            ("updates.jsonl", "updates_offset", "train_step", False),
            ("validation.jsonl", "validation_offset", "validation_step", True),
        ):
            rows, offset = read_rows(root / filename, state[offset_key])
            events.extend((r["step"], is_validation, r) for r in rows)
            state[offset_key] = offset
            if rows:
                state[step_key] = rows[-1]["step"]
        for step, is_validation, row in sorted(events, key=lambda item: item[0]):
            values = metrics(row, is_validation)
            if not is_validation and time.monotonic() - last_resources_at >= 30:
                values.update(resource_metrics(root, config["storage"]))
                last_resources_at = time.monotonic()
            swanlab.log(values, step=step)
        state["updated_at_unix"] = time.time()
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(state_path)
        if finished:
            receipt = json.loads(exit_path.read_text())
            status = json.loads((root / "status.json").read_text())
            if receipt["native_exit"] == 0 and status["status"] == "completed":
                swanlab.finish()
            else:
                swanlab.finish(state="crashed", error=json.dumps(receipt))
            return
        time.sleep(5)


if __name__ == "__main__":
    main()
