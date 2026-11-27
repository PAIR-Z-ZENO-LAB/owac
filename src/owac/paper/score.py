"""Score a complete pair of fixed and switching LIBERO runs for one checkpoint."""

import argparse
import gzip
import json
from pathlib import Path

from owac.runtime.dynamic_results import aggregate as switching_summary
from owac.runtime.isp_results import aggregate as fixed_summary
from owac.simulation.random_lighting import PROTOCOL, build_manifest, episode_identity
from owac.simulation.rgb_observation import RAW_PROTOCOL


def receipts(directory, expected, *, switching=False):
    rows, seen, checkpoints = [], set(), set()
    journals = sorted([*directory.rglob("episodes.jsonl"), *directory.rglob("episodes.jsonl.gz")])
    for path in journals:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as stream:
            for line in stream:
                row = json.loads(line)
                key = row["id"]
                if key in seen or key not in expected:
                    raise ValueError(f"Duplicate or unexpected episode {key}")
                if (
                    not row["formal"]
                    or row["native_exit"] != 0
                    or row["observation_protocol"] != RAW_PROTOCOL
                ):
                    raise ValueError(f"Invalid formal receipt {key}")
                seen.add(key)
                checksum = row["checkpoint_sha256"]
                if (
                    not isinstance(checksum, str)
                    or len(checksum) != 64
                    or any(c not in "0123456789abcdef" for c in checksum)
                ):
                    raise ValueError(f"Invalid checkpoint checksum for {key}")
                checkpoints.add(checksum)
                entry = expected[key]
                if (row["suite"], row["task"], row["initial_index"]) != (
                    entry["suite"],
                    entry["task_id"],
                    entry["init_state_index"],
                ):
                    raise ValueError(f"Incorrect initial state {key}")
                if switching:
                    if row.get("illumination_protocol") != PROTOCOL or any(
                        row[k] != entry[k] for k in ("static_id", "schedule_id", "schedule_sha256")
                    ):
                        raise ValueError(f"Switching schedule differs for {key}")
                    rows.append(
                        {
                            "static_id": row["static_id"],
                            "schedule_id": row["schedule_id"],
                            "scored_success": row["common_budget_success"],
                            "success": row["success"],
                            "switches": row["switch_count"]
                            if "switch_count" in row
                            else len(row["applied_lighting_events"]),
                        }
                    )
                else:
                    if row["lighting"] != entry["lighting_domain"]:
                        raise ValueError(f"Incorrect fixed lighting {key}")
                    rows.append(
                        {
                            k: row[k]
                            for k in (
                                "id",
                                "suite",
                                "task",
                                "initial_index",
                                "lighting",
                                "formal",
                                "native_exit",
                                "success",
                                "common_budget_success",
                            )
                        }
                    )
    if seen != set(expected) or len(checkpoints) != 1:
        raise ValueError(
            f"Expected {len(expected)} episodes from one checkpoint, found {len(rows)}"
        )
    return rows, checkpoints.pop()


def score(fixed, switching, manifest):
    fixed_rows, fixed_hash = receipts(fixed, {episode_identity(e): e for e in manifest["entries"]})
    entries = build_manifest(manifest)["entries"]
    dynamic, dynamic_hash = receipts(
        switching, {e["dynamic_id"]: e for e in entries}, switching=True
    )
    if dynamic_hash != fixed_hash:
        raise ValueError("Fixed and switching runs have different checkpoints")
    fixed_result = fixed_summary(fixed_rows, len(entries), success_key="common_budget_success")
    normal = {
        r["id"]: {"success": r["common_budget_success"]}
        for r in fixed_rows
        if r["lighting"] == "Normal"
    }
    switching_result = switching_summary(dynamic, normal, len(entries))
    return {
        "checkpoint_sha256": fixed_hash,
        "fixed": fixed_result,
        "all_five_pct": 100 * fixed_result["all_five_successes"] / fixed_result["paired_count"],
        "switching": switching_result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixed", "switching", "manifest"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            score(args.fixed, args.switching, json.loads(args.manifest.read_text())), indent=2
        )
    )


if __name__ == "__main__":
    main()
