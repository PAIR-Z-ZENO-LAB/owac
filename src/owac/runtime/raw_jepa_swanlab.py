"""Upload JEPA scalar events and resource metrics to the named SwanLab run."""

import argparse
import json
import time
from pathlib import Path

from owac.runtime.raw_action_swanlab import read_rows, resource_metrics


def main():
    import swanlab

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root
    config = json.loads((root / "config.json").read_text())
    tracking = config["tracking"]
    state_path = root / "swanlab-state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"offset": 0, "step": 0}
    swanlab.login(api_key=args.key_file.read_text().strip(), save=False)
    run = swanlab.init(
        project=tracking["project"],
        name=tracking["name"],
        id=tracking["run_id"],
        group=tracking["group"],
        job_type=tracking["job_type"],
        resume="allow",
        mode="online",
        public=False,
        config=config,
        logdir=str(root / "swanlab"),
        tags=tracking.get(
            "tags", ["stage:J1", "data:libero", "lighting:five", "objective:jepa", "target:100k"]
        ),
        settings=swanlab.Settings(
            probe={
                key: False
                for key in (
                    "hardware",
                    "runtime",
                    "requirements",
                    "conda",
                    "git",
                    "swanlab",
                    "monitor",
                )
            },
            terminal={"proxy_type": "none"},
        ),
    )
    state["url"] = run.url
    last_resources = 0
    while True:
        rows, offset = read_rows(root / "metrics.jsonl", state["offset"])
        for row in rows:
            values = row["metrics"]
            if time.monotonic() - last_resources >= 30:
                values.update(resource_metrics(root, config["storage"]))
                last_resources = time.monotonic()
            swanlab.log(values, step=row["step"])
            state["step"] = row["step"]
        state["offset"] = offset
        state["updated_at"] = time.time()
        temp = state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(state, indent=2) + "\n")
        temp.replace(state_path)
        receipt_path = root / "train.exit.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt["native_exit"] == 0:
                swanlab.finish()
            else:
                swanlab.finish(state="crashed", error=json.dumps(receipt))
            return
        time.sleep(5)


if __name__ == "__main__":
    main()
