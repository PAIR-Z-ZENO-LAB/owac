"""Compare private SwanLab curves against every scalar journal and final replay."""

import json
import math
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

from prepare_pouring import read, write


def expected(path):
    curves = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        for key, value in (row.get("values") or row["metrics"]).items():
            if isinstance(value, (int, float)):
                curves.setdefault(key, {})[row["step"]] = value
    return curves


def compare(run, local):
    keys = sorted(local)
    for start in range(0, len(keys), 10):
        batch = keys[start : start + 10]
        use_full = any(len(local[key]) > 1500 for key in batch)
        data = run.metrics(batch, all=use_full)["list"]
        cloud = {
            series["key"]: {point["index"]: point["data"] for point in series["metrics"]}
            for series in data
        }
        for key in batch:
            for step, value in local[key].items():
                remote = cloud.get(key, {}).get(step)
                if remote is None or not math.isclose(
                    value, float(remote), rel_tol=1e-5, abs_tol=1e-7
                ):
                    raise ValueError(f"SwanLab differs: {key} at {step}")


def main(root):
    import swanlab

    api = swanlab.Api(api_key=os.environ["SWANLAB_API_KEY"])
    destination = (os.environ["SWANLAB_WORKSPACE"], os.environ.get("SWANLAB_PROJECT", "OWAC"))
    if api.project("/".join(destination)).visibility != "PRIVATE":
        raise ValueError("Expected private OWAC project")
    spec = read(root / "source/configs/experiments/real_pouring_20261123.json")
    results = {}
    for name in ("vision", "owac", "dino-corp", "rawvla", "rawild"):
        folder = root / ("vision" if name == "vision" else f"runs/{name}")
        url = read(folder / "swanlab.json")["url"]
        workspace, project, _, run_id = urlsplit(url).path.strip("/").lstrip("@").split("/")
        if (workspace, project) != destination:
            raise ValueError(f"Unexpected destination: {url}")
        local = expected(folder / "metrics.jsonl")
        final = spec["visual_updates" if name == "vision" else "action_updates"]
        key = "visual/all/shift_and_minus1ev_top1" if name == "vision" else "replay/normalized_l1"
        if final not in local.get(key, {}):
            raise ValueError(f"Incomplete local journal: {name} {key} {final}")
        for attempt in range(3):
            try:
                run = api.run(f"{workspace}/{project}/{run_id}")
                if run.state != "FINISHED":
                    raise ValueError(f"SwanLab {name}: {run.state}")
                compare(run, local)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(12)
        results[name] = dict(
            url=url,
            state=run.state,
            scalar_series=len(local),
            scalar_points=sum(map(len, local.values())),
            final_step=final,
            final_metric=local[key][final],
        )
        if name == "vision":
            series = run.medias(["examples/U2", "examples/U3"], all=True)["list"]
            actual = {
                item["key"]: {point["index"] for point in item["metrics"] if point.get("items")}
                for item in series
            }
            steps = set(read(root / "configs/vision.json")["evaluation"]["panel_steps"])
            if any(
                not steps <= actual.get(camera, set()) for camera in ("examples/U2", "examples/U3")
            ):
                raise ValueError("Missing held-out camera panels on SwanLab")
        else:
            replay = read(folder / "replay/results.json")
            validation = {
                name
                for name, (part, _) in read(root / "receipts/split.json")["episodes"].items()
                if part == "validation"
            }
            if set(replay["validation"]) != validation:
                raise ValueError(f"Incomplete held-out replay: {name}")
            results[name]["selected_step"] = replay["step"]
        print(name, results[name], flush=True)
    write(root / "receipts/swanlab-final.json", results)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
