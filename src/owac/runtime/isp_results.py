"""Publish only observed formal closed-loop results to one evaluation-only run."""

import argparse
import json
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

LIGHTS = ["Normal", "Low", "ExtremeLow", "Over", "ExtremeOver"]
TABLE_KEYS = ("formal/六组对照表（最新）", "formal/逐任务逐光照（最新）")


def aggregate(rows, expected=10000, *, success_key="success"):
    identities = set()
    groups, paired = defaultdict(list), defaultdict(dict)
    for row in rows:
        if not row.get("formal") or row.get("native_exit") != 0:
            raise ValueError("Only native formal rollouts belong in the formal run")
        if row["id"] in identities:
            raise ValueError(f"Duplicate rollout: {row['id']}")
        identities.add(row["id"])
        groups[(row["suite"], row["lighting"])].append(row)
        paired[(row["suite"], row["task"], row["initial_index"])][row["lighting"]] = bool(
            row.get(success_key, row["success"])
        )
    counts = {
        light: {
            "episodes": sum(len(v) for (_, name), v in groups.items() if name == light),
            "successes": sum(
                sum(r.get(success_key, r["success"]) for r in v)
                for (_, name), v in groups.items()
                if name == light
            ),
        }
        for light in LIGHTS
    }
    rates = {
        key: 100 * value["successes"] / value["episodes"]
        for key, value in counts.items()
        if value["episodes"]
    }
    complete_pairs = [value for value in paired.values() if set(value) == set(LIGHTS)]
    complete = len(rows) == expected and all(
        v["episodes"] == expected // 5 for v in counts.values()
    )
    return {
        "episodes": len(rows),
        "expected": expected,
        "complete": complete,
        "successes": sum(row.get(success_key, row["success"]) for row in rows),
        "lighting": counts,
        "mean_pct": statistics.mean(rates.values()) if len(rates) == 5 else None,
        "std_pp": statistics.pstdev(rates.values()) if len(rates) == 5 else None,
        "worst_pct": min(rates.values()) if len(rates) == 5 else None,
        "paired_count": len(complete_pairs),
        "all_five_successes": sum(all(v.values()) for v in complete_pairs),
    }


def collect_rows(cfg, root):
    """Combine durable evaluation receipts without importing training or partial lines."""
    rows_by_method = {name: [] for name in cfg.get("expected_methods", [])}
    roots = [root / "evaluation", *map(Path, cfg.get("additional_evaluation_roots", []))]
    for directory in roots:
        for path in sorted(directory.glob("*/shard-*/episodes.jsonl")):
            method = path.parent.parent.name
            if cfg.get("expected_methods") and method not in rows_by_method:
                raise ValueError(f"Unexpected formal method: {method}")
            with path.open() as stream:
                for line in stream:
                    if line.endswith("\n"):
                        row = json.loads(line)
                        metadata = cfg.get("method_metadata", {}).get(method, {})
                        protocol = metadata.get("protocol")
                        if protocol and row.get("observation_protocol") != protocol:
                            raise ValueError(f"Mixed observation protocols for {method}")
                        checkpoint = metadata.get("checkpoint_sha256")
                        if checkpoint and row.get("checkpoint_sha256") != checkpoint:
                            raise ValueError(f"Mixed policy checkpoints for {method}")
                        rows_by_method.setdefault(method, []).append(row)
    return rows_by_method


def publish(cfg, root):
    import swanlab

    summaries = {}
    rows_by_method = collect_rows(cfg, root)
    for method, rows in rows_by_method.items():
        summaries[method] = aggregate(rows)
        if cfg.get("report_common_budget"):
            summaries[method]["common_budget"] = aggregate(
                rows, success_key="common_budget_success"
            )
    root.joinpath("formal-summary.json").write_text(json.dumps(summaries, indent=2))
    if not summaries:
        return None
    api = swanlab.Api(api_key=Path("/root/.config/owac/swanlab-api-key").read_text().strip())
    run = api.run(f"Jeong/OWAC/{cfg['swanlab']['id']}")
    client = api._ctx.client
    timestamp = datetime.now(UTC).isoformat()
    expected_count = len(cfg.get("expected_methods", range(6)))
    complete = len(summaries) == expected_count and all(v["complete"] for v in summaries.values())
    if not complete and run.state != "RUNNING":
        info = client.get(f"project/Jeong/OWAC/runs/{run.run_id}").data
        resumed = client.post(
            "project/Jeong/OWAC/experiment",
            data={
                "cuid": cfg["swanlab"]["id"],
                "name": cfg["swanlab"]["name"],
                "colors": info["colors"],
                "createdAt": timestamp,
                "cluster": cfg["swanlab"]["group"],
                "job": cfg["swanlab"]["job_type"],
            },
        ).data
        if resumed["cuid"] != run.run_id:
            raise RuntimeError("The current tables must stay in the existing formal run")
    headers = [
        "方法",
        "输入路线",
        "策略实现",
        "表中评分步数(S/O/G/L)",
        "控制 / 步数上限(S/O/G/L)",
        "完成场数",
        "计划场数",
        "状态",
        *LIGHTS,
        "五光平均 %",
        "光照标准差 pp",
        "最弱光照 %",
        "五光配对数",
        "五光全成功数",
        "原版步数上限成功/已评估",
    ]
    table_rows = []
    for method, summary in summaries.items():
        native_summary = summary
        if cfg.get("report_common_budget"):
            summary = summary["common_budget"]
        metadata = cfg.get("method_metadata", {}).get(method, {})
        cells = [
            f"{summary['lighting'][light]['successes']}/{summary['lighting'][light]['episodes']}"
            for light in LIGHTS
        ]
        table_rows.append(
            [
                method,
                metadata.get("route", "原版 RGB" if method.endswith("-rgb") else "RAW→ISP→策略"),
                metadata.get("implementation", "—"),
                "220/280/300/520" if cfg.get("report_common_budget") else "各策略原版",
                metadata.get(
                    "control",
                    "绝对 / 800/800/800/900" if method == "xvla-rgb" else "增量 / 220/280/300/520",
                ),
                summary["episodes"],
                summary["expected"],
                "完整" if summary["complete"] else ("进行中" if summary["episodes"] else "待开始"),
                *cells,
                *[
                    round(summary[key], 3) if summary[key] is not None else "—"
                    for key in ("mean_pct", "std_pp", "worst_pct")
                ],
                summary["paired_count"],
                summary["all_five_successes"],
                f"{native_summary['successes']}/{native_summary['episodes']}",
            ]
        )
    table = swanlab.echarts.Table()
    if cfg.get("paper_table"):
        from owac.runtime.paper_results import build_paper_table

        headers, table_rows = build_paper_table(cfg["paper_table"], summaries)
    table.add(headers, table_rows)
    detailed = swanlab.echarts.Table()
    detail_rows = []
    for method, rows in rows_by_method.items():
        groups = defaultdict(list)
        for row in rows:
            groups[(row["suite"], row["task"], row["lighting"])].append(row)
        for (suite, task, light), group in sorted(groups.items()):
            native_successes = sum(row["success"] for row in group)
            successes = sum(
                row.get("common_budget_success", row["success"])
                if cfg.get("report_common_budget")
                else row["success"]
                for row in group
            )
            detail_rows.append(
                [
                    method,
                    suite,
                    task,
                    light,
                    successes,
                    len(group),
                    round(100 * successes / len(group), 3),
                    native_successes,
                ]
            )
    detailed.add(
        [
            "方法",
            "Suite",
            "任务",
            "光照",
            "成功",
            "已评估",
            "成功率 %",
            "原版步数上限成功",
        ],
        detail_rows,
    )
    bodies = [value.dump_options().encode("utf-8") for value in (table, detailed)]
    if cfg.get("paper_table"):
        from owac.runtime.paper_results import format_paper_columns

        bodies[0] = json.dumps(
            format_paper_columns(json.loads(bodies[0]), cfg["paper_table"], summaries),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    return upload_current_tables(
        cfg,
        root,
        api,
        run,
        bodies,
        complete=complete,
        episode_count=sum(s["episodes"] for s in summaries.values()),
        methods=list(summaries),
        paper_rows=len(table_rows) if cfg.get("paper_table") else None,
    )


def upload_current_tables(
    cfg,
    root,
    api,
    run,
    bodies,
    *,
    complete,
    episode_count,
    methods,
    table_keys=TABLE_KEYS,
    paper_rows=None,
    stopped=False,
):
    """Overwrite two fixed resources, attaching each exactly once at step zero."""
    import requests

    client = api._ctx.client
    timestamp = datetime.now(UTC).isoformat()
    paths = [f"media/echarts/formal-latest-{name}.json" for name in ("summary", "details")]
    current_dir = root / "current-tables"
    current_dir.mkdir(exist_ok=True)
    for name, body in zip(("summary", "details"), bodies, strict=True):
        (current_dir / f"{name}.json").write_bytes(body)
    urls = client.post(
        "resources/presigned/put", data={"experimentId": run.run_id, "paths": paths}
    ).data["urls"]
    for url, body in zip(urls, bodies, strict=True):
        try:
            response = requests.put(
                url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Cache-Control": "no-store, max-age=0",
                },
                timeout=30,
            )
        except requests.RequestException as error:
            raise RuntimeError(f"Table resource upload failed: {type(error).__name__}") from None
        if not response.ok:
            raise RuntimeError(f"Table resource upload returned HTTP {response.status_code}")
    # Bind each table once. Repeated logs at the same step can accumulate media
    # attachments; overwriting these two resources keeps exactly one table per
    # key and one step, so SwanLab does not render its history slider.
    binding_path = Path(cfg.get("table_binding", current_dir / "binding.json"))
    if binding_path.exists():
        binding = json.loads(binding_path.read_text())
        if binding["run_id"] != run.run_id or binding["keys"] != list(table_keys):
            raise ValueError("Existing table binding belongs to a different run or metric")
    if not binding_path.exists():
        client.post(
            "projects/Jeong/OWAC/series",
            data={
                "series": [
                    {"key": key, "type": "ECHARTS", "sectionName": "正式闭环验证", "hidden": True}
                    for key in table_keys
                ]
            },
        )
        client.post(
            "house/metrics",
            data={
                "type": "media",
                "projectId": run.project_id,
                "experimentId": run.run_id,
                "metrics": [
                    {"key": key, "index": 0, "data": [path], "more": [{}], "create_time": timestamp}
                    for key, path in zip(table_keys, paths, strict=True)
                ],
            },
        )
        binding_path.write_text(
            json.dumps({"run_id": run.run_id, "keys": table_keys, "paths": paths, "step": 0})
        )
    if complete or stopped:
        client.put(
            f"project/Jeong/OWAC/runs/{run.run_id}/state",
            data={
                "state": "FINISHED" if complete else "ABORTED",
                "finishedAt": timestamp,
                "from": "sdk",
            },
        )
    else:
        client.post(f"house/experiments/{run.run_id}/heartbeat", data={"flagId": "123456"})
    (root / "swanlab-upload.json").write_text(
        json.dumps(
            {
                "run_id": cfg["swanlab"]["id"],
                "results_url": cfg["swanlab"].get("results_url"),
                "episodes": episode_count,
                "stopped_by_user": stopped,
                "methods": methods,
                "table_mode": "current_only",
                "table_step": 0,
                "table_keys": list(table_keys),
                "table_paths": paths,
                "submitted_at": timestamp,
                "presentation": "paper" if cfg.get("paper_table") else "experiment_monitor",
                "paper_rows": paper_rows,
                "missing_value": cfg.get("paper_table", {}).get("missing_value"),
            }
        )
    )
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    publish(cfg, Path(cfg["root"]))


if __name__ == "__main__":
    main()
