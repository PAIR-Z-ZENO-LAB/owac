"""Validation-only current tables for paired random illumination rollouts."""

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from owac.runtime.isp_results import upload_current_tables
from owac.simulation.random_lighting import PROTOCOL

TABLE_KEYS = ("dynamic/随机换光对照表（最新）", "dynamic/逐任务逐序列（最新）")


def collect(cfg):
    manifest = json.loads(Path(cfg["manifest"]).read_text())
    identities = {e["dynamic_id"]: e for e in manifest["entries"]}
    methods = {method: [] for method in cfg["methods"]}
    for method, item in cfg["methods"].items():
        seen = set()
        directory = Path(item["root"]) / "evaluation" / method
        for path in sorted(directory.glob("shard-*/episodes.jsonl")):
            with path.open() as stream:
                for line in stream:
                    if not line.endswith("\n"):
                        continue
                    row = json.loads(line)
                    if not row.get("formal") or row.get("native_exit") != 0:
                        raise ValueError("Only native formal validation receipts are accepted")
                    if row.get("illumination_protocol") != PROTOCOL:
                        raise ValueError("Static or different illumination protocol in dynamic run")
                    entry = identities[row["id"]]
                    if row["id"] in seen:
                        raise ValueError(f"Duplicate dynamic identity: {row['id']}")
                    if row["schedule_sha256"] != entry["schedule_sha256"]:
                        raise ValueError("Mismatched frozen random schedule")
                    if row["static_id"] != entry["static_id"]:
                        raise ValueError("Mismatched physical initial state")
                    if row["schedule_id"] != entry["schedule_id"]:
                        raise ValueError("Mismatched schedule replicate")
                    for field in ("observation_protocol", "checkpoint_sha256"):
                        expected = item.get(field)
                        if expected and row.get(field) != expected:
                            raise ValueError(f"Mixed {field}: {method}")
                    seen.add(row["id"])
                    # The publisher keeps bounded scalar receipts, not images or event arrays.
                    methods[method].append(
                        {
                            k: row[k]
                            for k in (
                                "id",
                                "static_id",
                                "suite",
                                "task",
                                "initial_index",
                                "schedule_id",
                                "success",
                                "steps",
                            )
                        }
                        | {
                            "scored_success": row.get("common_budget_success", row["success"]),
                            "switches": len(row["applied_lighting_events"]),
                        }
                    )
    return methods


def aggregate(rows, reference, expected):
    paired = defaultdict(dict)
    successes, normal_successes, switches = 0, 0, 0
    for row in rows:
        baseline = reference[row["static_id"]]
        successes += bool(row["scored_success"])
        normal_successes += bool(baseline["success"])
        switches += row["switches"]
        group = paired[row["static_id"]]
        if row["schedule_id"] in group:
            raise ValueError("Duplicate sequence within a physical initial state")
        group[row["schedule_id"]] = bool(row["scored_success"])
    complete_pairs = [p for p in paired.values() if set(p) == set(range(5))]
    n = len(rows)
    complete = n == expected and len(complete_pairs) * 5 == expected
    ci = None
    if complete:
        # Resample physical initial states within each task, retaining all five
        # schedules together. These are evaluation CIs, not training-seed variance.
        tasks = defaultdict(list)
        for identity, group in paired.items():
            tasks["/".join(identity.split("/")[:2])].append(np.mean(list(group.values())))
        rng = np.random.default_rng(20261115)
        sampled = np.zeros(2000)
        for group in tasks.values():
            values = np.asarray(group)
            sampled += values[rng.integers(len(values), size=(2000, len(values)))].sum(axis=1)
        ci = (100 * np.quantile(sampled / len(paired), [0.025, 0.975])).tolist()
    return {
        "episodes": n,
        "expected": expected,
        "complete": complete,
        "successes": successes,
        "success_pct": 100 * successes / n if n else None,
        "matched_normal_pct": 100 * normal_successes / n if n else None,
        "drop_pp": 100 * (normal_successes - successes) / n if n else None,
        "complete_initial_states": len(complete_pairs),
        "all_sequences_successes": sum(all(p.values()) for p in complete_pairs),
        "switches": switches,
        "confidence_interval": ci,
        "native_successes": sum(r["success"] for r in rows),
    }


def tables(cfg, rows_by_method, references):
    import swanlab

    summaries, table_rows, details = {}, [], []
    for method, rows in rows_by_method.items():
        item = cfg["methods"][method]
        summary = aggregate(rows, references[method], cfg["episodes_per_method"])
        summaries[method] = summary

        def rate(key, values=summary):
            return round(values[key], 2) if values[key] is not None else "—"

        ci = summary["confidence_interval"]
        table_rows.append(
            [
                item["group"],
                item["label"],
                f"{summary['episodes']}/{summary['expected']}",
                "完整" if summary["complete"] else "进行中" if rows else "待开始",
                rate("success_pct"),
                "—" if ci is None else f"[{ci[0]:.2f}, {ci[1]:.2f}]",
                rate("matched_normal_pct"),
                rate("drop_pp"),
                f"{summary['all_sequences_successes']}/{summary['complete_initial_states']}",
                summary["switches"],
                f"{summary['native_successes']}/{summary['episodes']}",
            ]
        )
        groups = defaultdict(list)
        for row in rows:
            groups[(row["suite"], row["task"], row["schedule_id"])].append(row)
        for (suite, task, sequence), group in sorted(groups.items()):
            success = sum(r["scored_success"] for r in group)
            normal = sum(references[method][r["static_id"]]["success"] for r in group)
            details.append(
                [
                    item["label"],
                    suite,
                    task,
                    sequence,
                    success,
                    len(group),
                    round(100 * success / len(group), 2),
                    normal,
                    sum(r["switches"] for r in group),
                    sum(r["success"] for r in group),
                ]
            )
    main = swanlab.echarts.Table()
    main.add(
        [
            "类别",
            "方法",
            "评测进度",
            "状态",
            "随机换光成功率 %",
            "95% CI",
            "配对正常光 %",
            "正常光到换光下降 pp",
            "五条序列均成功 / 完整初态",
            "实际换光次数",
            "原版完整步数成功/已评估",
        ],
        table_rows,
    )
    detail = swanlab.echarts.Table()
    detail.add(
        [
            "方法",
            "Suite",
            "任务",
            "随机序列",
            "成功",
            "已评估",
            "成功率 %",
            "配对正常光成功",
            "实际换光次数",
            "原版完整步数成功",
        ],
        details,
    )
    bodies = []
    for table_index, table in enumerate((main, detail)):
        payload = json.loads(table.dump_options())
        labels = [column["field"] for column in payload["colDefs"]]
        fields = [f"column_{index}" for index in range(len(labels))]
        payload["rowData"] = [
            {field: row[label] for field, label in zip(fields, labels, strict=True)}
            for row in payload["rowData"]
        ]
        for index, (column, field, label) in enumerate(
            zip(payload["colDefs"], fields, labels, strict=True)
        ):
            width = 235 if index == (1 if table_index == 0 else 0) else 150
            column.update(
                field=field,
                headerName=label,
                width=width,
                minWidth=width,
                resizable=True,
                wrapHeaderText=True,
                autoHeaderHeight=True,
                headerTooltip=cfg["caption"],
            )
            if index < (2 if table_index == 0 else 1):
                column["pinned"] = "left"
        if table_index == 0:
            for row, summary in zip(payload["rowData"], summaries.values(), strict=True):
                row["success_support"] = (
                    f"成功 {summary['successes']} / 已评估 {summary['episodes']}"
                )
            payload["colDefs"][4]["tooltipField"] = "success_support"
        payload["caption"] = cfg["caption"]
        bodies.append(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode())
    return summaries, bodies


def publish(cfg):
    import swanlab

    root = Path(cfg["root"])
    rows = collect(cfg)
    references = json.loads(Path(cfg["static_reference"]).read_text())
    summaries, bodies = tables(cfg, rows, references)
    (root / "formal-summary.json").write_text(json.dumps(summaries, indent=2))
    complete = all(s["complete"] for s in summaries.values())
    api = swanlab.Api(api_key=Path("/root/.config/owac/swanlab-api-key").read_text().strip())
    run = api.run(f"Jeong/OWAC/{cfg['swanlab']['id']}")
    if not complete and run.state != "RUNNING":
        client = api._ctx.client
        info = client.get(f"project/Jeong/OWAC/runs/{run.run_id}").data
        resumed = client.post(
            "project/Jeong/OWAC/experiment",
            data={
                "cuid": cfg["swanlab"]["id"],
                "name": cfg["swanlab"]["name"],
                "colors": info["colors"],
                "createdAt": datetime.now(UTC).isoformat(),
                "cluster": cfg["swanlab"]["group"],
                "job": cfg["swanlab"]["job_type"],
            },
        ).data
        if resumed["cuid"] != run.run_id:
            raise RuntimeError("Unexpected validation run binding")
    upload_current_tables(
        cfg,
        root,
        api,
        run,
        bodies,
        complete=complete,
        episode_count=sum(s["episodes"] for s in summaries.values()),
        methods=list(summaries),
        table_keys=TABLE_KEYS,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    publish(json.loads(args.config.read_text()))


if __name__ == "__main__":
    main()
