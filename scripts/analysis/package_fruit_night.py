"""Pair a selected training export with a calibrated RTC adapter and recorded-data replay."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from replay_fruit_rtc import replay

from owac.runtime.fruit_live import Travel
from owac.runtime.fruit_rtc import PrefixRegression


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--id", dest="candidate", help="Menu ID when exporting another step of one run")
    p.add_argument("--label", help="Operator label for the additional checkpoint")
    args = p.parse_args()
    root = args.root
    run = root / "runs" / args.variant
    evaluation = root / "evaluation" / args.variant
    scores = json.loads((evaluation / "scores.json").read_text())[args.stage]
    step = scores["step"]
    folder = root / "release" / (args.candidate or args.variant)
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = run / f"policy-{args.stage}.pt"
    if not (folder / "policy.pt").exists():
        os.link(checkpoint, folder / "policy.pt")
    adapter = folder / "rtc.json"
    if not adapter.exists():
        temp = folder / "rtc.pending.json"
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("fit_fruit_rtc.py")),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(temp),
            ],
            check=True,
        )
        os.replace(temp, adapter)
    rtc = PrefixRegression.load(adapter, folder / "policy.pt", step)
    with np.load(evaluation / f"step-{step}.npz") as archive:
        traces = dict(archive)
    travel = Travel(root / "urdf", 0.2)
    replayed = [replay(traces, rtc, travel, delay) for delay in (0.012, 0.08)]
    cfg = json.loads((run / "config.json").read_text())
    published = dict(
        id=folder.name,
        variant=args.variant,
        label=args.label,
        stage=args.stage,
        step=step,
        sha256=scores["sha256"],
        source=str(checkpoint),
        parent=cfg["continuation"],
        replay=replayed,
    )
    temp = folder / "release.pending.json"
    temp.write_text(json.dumps(published, indent=2, allow_nan=False) + "\n")
    os.replace(temp, folder / "release.json")


if __name__ == "__main__":
    main()
