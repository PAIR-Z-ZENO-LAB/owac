"""Compare completed policies with the inter-chunk waits observed in two real OWAC attempts."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from owac.raw.all_bench_distillation import load_inventory
from owac.training.fruit_grasp import NeroHand
from owac.training.teleop_recovery import fixed_language, load_export
from owac.training.teleop_recovery_replay import RecoveryBank, score_replay


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment", type=Path)
    p.add_argument("urdf", type=Path)
    args = p.parse_args()
    root = args.experiment
    torch.set_num_threads(2)
    src = [s for s in load_inventory(root / "aligned/inventory.json")[0] if s.split == "validation"]
    track = json.loads((root / "field-tracking.json").read_text())
    hand = NeroHand(args.urdf, "cpu")
    cmd = next((root.parent / "grasp-comparison-20261124/field").glob("commands-*.jsonl"))
    gaps = [[1]]
    for k, items in itertools.groupby(
        map(json.loads, cmd.read_text().splitlines()), key=lambda r: r["policy"]
    ):
        if k != "owac":
            continue
        rows = list(items)
        first = [i for i, r in enumerate(rows) if r["row"] == 0 and i > 0]
        gaps.append(
            [
                max(0, round((rows[i]["time_ns"] - rows[i - 1]["time_ns"]) * 30 / 1e9) - 1)
                for i in first
            ]
        )
    models = {
        "original": (root.parent / "runs/owac/policy-best.pt", [12]),
        "absolute": (root / "absolute/policy-best.pt", [5]),
        "spatial": (root / "spatial-action/policy-best.pt", [4, 5, 6, 8, 12]),
        "history": (root / "motion-history/policy-best.pt", [4, 5, 6, 8, 12]),
    }
    results = {}
    for name, (path, horizons) in models.items():
        model, _ = load_export(path)
        text = fixed_language(model)
        if not results:
            bank = RecoveryBank(src, root / "visual", model.cfg, track)
            bank.load_spatial(root / "spatial")
        bank.configure_output(model.cfg)
        if hasattr(model, "correct"):
            bank.history_rows(model.cfg["model"]["state_history_lags"])
        for n in horizons:
            runs = []
            for gap in gaps:
                errors = bank.rollout(model, text, n, gaps=gap)
                summary, episodes = score_replay(bank, errors, hand)
                runs.append(dict(summary=summary, episodes=episodes))
            keys = (
                "first_grasp_xy_mm",
                "four_events_xyz_mm",
                "four_events_xy_mm",
                "full_hand_mm",
                "full_gripper_mm",
                "four_events_timing_mae_s",
            )
            means = {k: float(np.mean([p["summary"][k] for p in runs])) for k in keys}
            event_count = sum(p["summary"]["events"] for p in runs)
            result = dict(
                policy=name,
                checkpoint=str(path),
                execution_horizon=n,
                events_across_schedules=event_count,
                means=means,
                protocols=runs,
            )
            results[f"{name}-{n}"] = result
            print(name, n, event_count, means, flush=True)
    best = min(
        results.values(),
        key=lambda r: (-r["events_across_schedules"], r["means"]["four_events_xyz_mm"]),
    )
    (root / "execution-comparison.json").write_text(
        json.dumps(
            dict(
                gaps=gaps,
                results=results,
                selected=best["policy"] + "-" + str(best["execution_horizon"]),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
