"""Read the latest pouring pause, its command history and accompanying RAW evidence."""

import argparse
import json
from pathlib import Path


def rows(path):
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            # The operator can open this report while the console is appending a row.
            if line.endswith("\n"):
                yield json.loads(line)


def report(folder):
    previous = {}
    latest = None
    for pause in rows(folder / "console-pauses.jsonl"):
        file = pause.get("command_file")
        stamp = pause.get("pause_time_ns")
        if not file or not stamp:
            continue
        since, sent, fault = previous.get(file, (0, 0, None))
        current = pause.get("pouring_fault")
        previous[file] = stamp, pause["sent"], current
        if pause["sent"] > sent or (current and current != fault):
            latest = pause, file, since, stamp
    if latest is None:
        return ["还没有已记录的 Pouring 候选试次暂停。"]
    event, file, start, stop = latest
    command = None
    count = 0
    for row in rows(folder / Path(file).name):
        if start < row["time_ns"] <= stop and row["policy"] == event["policy"]:
            command = row
            count += 1
    message = [
        f"{event['time']}  {event['policy']}  {event['reason']}",
        f"阶段 {event.get('pouring_phase')}，本次暂停前记录 {count} 条指令",
    ]
    if command:
        for label, key, stamp in (
            ("模型", "model_rad_m", "camera_wall_ns"),
            ("已发", "issued_rad_m", "time_ns"),
            ("最近 mux", "latest_mux_rad_m", "latest_mux_ns"),
            ("最近反馈", "feedback_rad_m", "feedback_ns"),
        ):
            if command.get(key) is not None:
                message.append(
                    f"{label}开度 {command[key][7] * 1000:.1f} mm  时间戳 {command.get(stamp)}"
                )
    if p := event.get("pause_frame"):
        message.append(f"暂停双相机画面 {p}")
    message += [f"原始指令 {folder / file}", "入碗量、洒落和杯底是否着桌，请核对视频或现场计分。"]
    return message


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    print("\n".join(report(parser.parse_args().folder)))
