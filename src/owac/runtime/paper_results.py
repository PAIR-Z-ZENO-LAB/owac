"""Project matched evaluation results, labeling optional live estimates explicitly."""

LIGHTS = ("Normal", "Low", "ExtremeLow", "Over", "ExtremeOver")
HEADERS = (
    "类别",
    "方法",
    "Normal (%) ↑",
    "Low (%) ↑",
    "ExtremeLow (%) ↑",
    "Over (%) ↑",
    "ExtremeOver (%) ↑",
    "Avg. (%) ↑",
    "Std. (pp) ↓",
    "Worst (%) ↑",
    "All-5 (%) ↑",
)
FIELDS = (
    "category",
    "method",
    "normal_pct",
    "low_pct",
    "extreme_low_pct",
    "over_pct",
    "extreme_over_pct",
    "mean_pct",
    "std_pp",
    "worst_pct",
    "all_five_pct",
)


def build_paper_table(spec, summaries):
    """Keep unavailable cells separate from measured zeros and partial estimates."""
    missing = spec["missing_value"]
    live = spec.get("show_partial", False)
    expected = spec["episodes_per_method"]
    per_light = expected // len(LIGHTS)
    rows = []
    sources = set()
    for method in spec["rows"]:
        source = method["source"]
        if source in sources:
            raise ValueError(f"Duplicate paper-table source: {source}")
        sources.add(source)
        summary = summaries.get(source)
        if summary is not None and summary["expected"] != expected:
            raise ValueError(f"Incompatible evaluation denominator for {source}")
        if summary is not None and spec["score"] == "common_budget":
            # Do not silently fall back to X-VLA's longer native action budget.
            summary = summary["common_budget"]
        metrics = [missing] * 9
        if summary is not None:
            for index, light in enumerate(LIGHTS):
                cell = summary["lighting"][light]
                if cell["episodes"] == per_light or (live and cell["episodes"]):
                    metrics[index] = round(100 * cell["successes"] / cell["episodes"], 2)
            if summary["complete"] or (
                live and all(summary["lighting"][light]["episodes"] for light in LIGHTS)
            ):
                metrics[5:8] = [
                    round(summary[key], 2) for key in ("mean_pct", "std_pp", "worst_pct")
                ]
            if (summary["complete"] and summary["paired_count"] == per_light) or (
                live and summary["paired_count"]
            ):
                metrics[8] = round(100 * summary["all_five_successes"] / summary["paired_count"], 2)
        identity = [method["group"], method["label"]]
        if live:
            count = summary["episodes"] if summary else 0
            status = (
                "完成" if summary and summary["complete"] else ("进行中" if count else "待开始")
            )
            identity.append(f"{status} {count}/{expected}")
        rows.append([*identity, *metrics])
    headers = list(HEADERS)
    if live:
        headers.insert(2, "评测进度")
    return headers, rows


def format_paper_columns(body, spec, summaries=None):
    """Use ordinary AG Grid column metadata in SwanLab's Table JSON format."""
    widths = [155, 210, 90, 80, 110, 80, 115, 90, 90, 90, 90]
    fields = list(FIELDS)
    if spec.get("show_partial"):
        widths.insert(2, 185)
        fields.insert(2, "progress")
    note = spec["caption"]
    # AG Grid interprets dots in field names as nested property access. Display
    # labels such as "Avg. (%)" must not double as the actual row-data keys.
    labels = [column["field"] for column in body["colDefs"]]
    body["rowData"] = [
        {field: row[label] for field, label in zip(fields, labels, strict=True)}
        for row in body["rowData"]
    ]
    if spec.get("show_partial") and summaries is not None:
        for row, method in zip(body["rowData"], spec["rows"], strict=True):
            summary = summaries.get(method["source"])
            if summary is None:
                continue
            if spec["score"] == "common_budget":
                summary = summary["common_budget"]
            for light, field in zip(LIGHTS, FIELDS[2:7], strict=True):
                cell = summary["lighting"][light]
                row[f"{field}_support"] = (
                    f"成功 {cell['successes']} / 已评估 {cell['episodes']}；"
                    f"计划 {spec['episodes_per_method'] // len(LIGHTS)} 场"
                )
            row["all_five_pct_support"] = (
                f"五光全成功 {summary['all_five_successes']} / "
                f"五光均已评估的初态 {summary['paired_count']}；"
                f"计划 {spec['episodes_per_method'] // len(LIGHTS)} 组"
            )
    for index, (column, width, field, label) in enumerate(
        zip(body["colDefs"], widths, fields, labels, strict=True)
    ):
        column.update(
            field=field,
            headerName=label,
            width=width,
            minWidth=width,
            resizable=True,
            sortable=False,
            wrapHeaderText=True,
            autoHeaderHeight=True,
            headerTooltip=note,
        )
        if index < 2:
            column.update(pinned="left")
        elif field != "progress":
            column.update(cellStyle={"textAlign": "right"})
        if spec.get("show_partial") and field in (*FIELDS[2:7], "all_five_pct"):
            column["tooltipField"] = f"{field}_support"
    body["caption"] = note
    return body
