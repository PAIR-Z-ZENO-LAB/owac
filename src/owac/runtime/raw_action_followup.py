"""Synchronize R1 evidence and research-page status while its owned job runs."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

BEGIN = "<!-- R1_LIVE_BEGIN -->"
END = "<!-- R1_LIVE_END -->"


def probe(host, port, remote):
    script = f"""
import json, subprocess
from pathlib import Path
root = Path({str(remote)!r})
def tail(name):
    path = root/name
    if not path.exists():
        return None
    line = subprocess.check_output(["tail", "-n", "1", str(path)], text=True).strip()
    return json.loads(line) if line else None
status = json.loads((root/"status.json").read_text())
exit_path = root/"train-r3.exit.json"
receipt = json.loads(exit_path.read_text()) if exit_path.exists() else None
if receipt is not None and receipt["native_exit"] != 0:
    status["status"] = "failed"
final = root/"final100000.pt"
print(json.dumps(dict(status=status, update=tail("updates.jsonl"),
    validation=tail("validation.jsonl"), receipt=receipt,
    final_bytes=final.stat().st_size if final.exists() else None)))
"""
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-p",
        str(port),
        host,
        "python -c " + shlex.quote(script),
    ]
    return json.loads(subprocess.check_output(command, text=True))


def update_documents(repo, evidence, remote):
    status, update, validation = evidence["status"], evidence["update"], evidence["validation"]
    step = update["step"] if update is not None else status["step"]
    state = status["status"]
    lines = [BEGIN, f"实际进度：**{step:,}/100,000更新**；状态：**{state}**。"]
    if update is not None:
        lines.append(f"最近训练batch flow loss：{update['flow_loss']:.8f}（第{step}步）。")
    if validation is not None:
        lines.append(
            f"最近验证：第{validation['step']}步；完整验证flow loss "
            f"{validation['groups']['all']['flow_loss']:.8f}。"
        )
    if state == "completed":
        lines.append(f"最终权重：`{remote}/final100000.pt`（{evidence['final_bytes']:,}字节）。")
        lines.append(
            "训练/验证曲线：`outputs/r1-raw-action-20261027/curves.png`、`curves.svg`；"
            "完整逐步记录同步到该本地目录。R2尚未执行。"
        )
    if evidence["receipt"] is not None:
        lines.append(f"原生退出码：{evidence['receipt']['native_exit']}。")
    lines.append(END)
    report = repo / "research/experiments/r1_raw_action_20261027.md"
    text = report.read_text()
    section = "\n".join(lines)
    if BEGIN in text:
        before, rest = text.split(BEGIN, 1)
        _, after = rest.split(END, 1)
        text = before + section + after
    else:
        heading, body = text.split("\n", 1)
        text = heading + "\n\n" + section + "\n" + body
    if state == "completed":
        text = text.replace(
            "R1正在训练，目标为100,000次实际优化器更新。尚未交付最终100k权重。",
            "R1已完成100,000次实际优化器更新，最终权重与完整曲线已保存。",
        )
    report.write_text(text)
    path = repo / "configs/research/hub.json"
    hub = json.loads(path.read_text())
    route = next(r for r in hub["routes"] if r["id"] == "R")
    label = {
        "running": "训练中",
        "completed": "100k已完成",
        "failed": "训练错误",
        "paused": "已暂停",
    }[state]
    route["status"] = f"R1{label} · {step:,}/100,000"
    route["stages"][0]["label"] = label
    if state == "completed":
        route["stages"][0]["state"] = "recorded"
        route["next"] = "R1最终100k权重已保存；R2闭环待执行。"
        route["stages"][0]["description"] = (
            "R1已完成100,000真实动作flow-matching更新；原LIBERO五照明源拆分，"
            "原始视觉初始化及重置stem/gate，语言/动作后端冻结。最终权重和曲线已保存。"
        )
    hub["strategy"]["boundary"] = f"R1{label}；{step:,}/100,000更新；闭环留给R2。"
    for node in hub["lineage"]["nodes"]:
        if node["id"] == "r":
            node["status"] = route["status"]
    path.write_text(json.dumps(hub, ensure_ascii=False, indent=2) + "\n")
    if state == "completed":
        for name in ("AGENTS.md", "README.md", "docs/status.md"):
            path = repo / name
            text = path.read_text()
            date = datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()
            if "R1已完成100,000" not in text.split("\n", 1)[0]:
                path.write_text(
                    f"{date} R1已完成100,000次真实动作监督更新，最终权重/曲线已保存；"
                    "详见 `research/experiments/r1_raw_action_20261027.md`。\n\n" + text
                )
    from owac.runtime.research_hub import build_snapshot, render_page

    snapshot = build_snapshot(repo)
    (repo / "outputs/research-hub/index.html").write_text(render_page(snapshot))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--remote", required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    output = args.repo / "outputs/r1-raw-action-20261027"
    output.mkdir(parents=True, exist_ok=True)
    last_validation = None
    while True:
        evidence = probe(args.host, args.port, args.remote)
        (output / "progress.json").write_text(json.dumps(evidence, indent=2) + "\n")
        validation = evidence["validation"]
        done = evidence["receipt"] is not None
        names = []
        if validation is not None and validation["step"] != last_validation:
            names += ["curves.png", "curves.svg"]
            last_validation = validation["step"]
        if done:
            names += [
                "updates.jsonl",
                "validation.jsonl",
                "config.json",
                "data-summary.json",
                "train-r3.launch.json",
                "train-r3.exit.json",
                "train-r3.stderr.log",
            ]
            # A failure before the first validation has no validation file.
            if validation is None:
                names.remove("validation.jsonl")
        for name in names:
            subprocess.run(
                [
                    "scp",
                    "-q",
                    "-P",
                    str(args.port),
                    f"{args.host}:{args.remote}/{name}",
                    str(output / name),
                ],
                check=True,
            )
        update_documents(args.repo, evidence, args.remote)
        if done or args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
