"""Select a fruit checkpoint, preview its live predictions and test it on the right NERO."""

import argparse
import curses
import fcntl
import json
import os
import signal
import sys
import time
from collections import Counter
from contextlib import ExitStack, suppress
from pathlib import Path
from threading import Event, Thread

from owac.runtime.fruit_trials import GraspTrials


def draw(screen, lines):
    screen.erase()
    h, w = screen.getmaxyx()
    for i, (label, highlight) in enumerate(lines[:h]):
        with suppress(curses.error):
            screen.addnstr(i, 0, label, max(0, w - 1), curses.A_BOLD if highlight else 0)
    screen.refresh()


def choose_weight(screen, choices, current):
    keys = list(choices)
    index = keys.index(current) if current in keys else 0
    screen.timeout(100)
    try:
        while True:
            height = max(1, screen.getmaxyx()[0] - 3)
            start = max(0, index - height + 1)
            lines = [("选择策略权重", True), ("↑ ↓ 选择    回车确认    Esc 返回", False)]
            for i in range(start, min(start + height, len(keys))):
                key = keys[i]
                lines.append(
                    (
                        f"{'›' if i == index else ' '} {i + 1}  {choices[key].label}    {key}",
                        i == index,
                    )
                )
            draw(screen, lines)
            value = screen.getch()
            if value in (27, ord("q")):
                return None
            if value in (curses.KEY_ENTER, 10, 13):
                return keys[index]
            if ord("1") <= value <= ord("9") and value - ord("1") < len(keys):
                return keys[value - ord("1")]
            if value in (curses.KEY_UP, ord("k")):
                index = (index - 1) % len(keys)
            if value in (curses.KEY_DOWN, ord("j")):
                index = (index + 1) % len(keys)
    finally:
        screen.nodelay(True)


def panel(screen, inputs, robot, speed, choices, trials=None, task="fruit"):
    import numpy as np

    from owac.runtime.fruit_live import row_at

    with suppress(curses.error):
        curses.curs_set(0)
    screen.nodelay(True)
    weight_keys = list(choices)
    next_draw = 0
    while True:
        now = time.monotonic()
        keys = []
        while (code := screen.getch()) != -1:
            if 0 <= code < 128:
                keys.append(chr(code).lower())
        if "q" in keys or "\x1b" in keys:
            return
        if " " in keys:
            robot.pause("已暂停")
        else:
            for key in keys:
                try:
                    if key in ("1", "2", "m"):
                        if len(weight_keys) == 1:
                            continue
                        robot.pause("选择权重，已暂停")
                        selected = (
                            choose_weight(screen, choices, inputs.key)
                            if key == "m"
                            else weight_keys[int(key) - 1]
                        )
                        if selected is not None:
                            if trials is not None:
                                trials.finish("unscored", time.monotonic(), robot.sent)
                            if inputs.key is not None:
                                robot.armed = False
                            if choices[selected].rtc is None:
                                robot.rtc_enabled = False
                            if selected != inputs.key and selected in robot.postprocessors:
                                robot.postprocessors[selected].reset()
                            inputs.select(selected)
                        if key == "m":
                            break
                    elif key == "c":
                        robot.connect()
                    elif key == "h":
                        if trials is not None:
                            trials.finish("unscored", now, robot.sent)
                        robot.start_reset(now)
                    elif key in ("s", "f") and trials is not None and trials.active:
                        robot.pause("本次已暂停，请摆好香蕉后开始下一次")
                        trials.finish(trials.success if key == "s" else "missed", now, robot.sent)
                    elif key == "t" and not robot.direct and trials is None:
                        robot.toggle_rtc()
                    elif key in ("n", "r") and "h" not in keys and "t" not in keys:
                        if key == "n" and trials is not None:
                            continue
                        sent_before = robot.sent
                        robot.step(now, continuous=key == "r")
                        if trials is not None and robot.running:
                            trials.begin(inputs.key, now, sent_before)
                except (OSError, RuntimeError, ValueError) as exc:
                    robot.pause(str(exc))
        robot.tick(now)
        if trials is not None and trials.active and now >= trials.next_sample:
            issued = robot.last_policy_target
            trials.observe(
                now,
                robot.sent,
                inputs.at("ROS", time.time_ns()),
                robot.last_model_target,
                issued[1] if issued and issued[0] == inputs.key else None,
                robot.last_mux,
            )
        if now >= next_draw or keys:
            next_draw = now + 0.1
            with inputs.condition:
                pred = inputs.prediction
                obs = inputs.observation
                receipts = list(inputs.receipts)
                trouble = "  ".join(f"{k} {v}" for k, v in inputs.events.items())
            state = inputs.at("ROS", time.time_ns()) or inputs.at("CAN", time.time_ns())
            fps = (
                (len(receipts) - 1) / (receipts[-1] - receipts[0])
                if (len(receipts) > 1 and receipts[-1] > receipts[0] and now - receipts[-1] < 1)
                else 0.0
            )
            step = choices[inputs.key].label if inputs.key else "未选择，可先复位"
            mode = (
                "正在复位"
                if robot.reset_deadline is not None
                else "连续执行"
                if robot.running
                else "已接通 / 等待执行"
                if robot.armed
                else "接通中"
                if robot.engaging
                else "只读预览"
            )
            detail = "等待 U2/U3 和右臂状态"
            if obs:
                seq = [f.sequence for f in obs.frames]
                detail = f"U2 #{seq[0]}   U3 #{seq[1]}   双路 {fps:.1f} Hz"
                detail += f"   画面距今 {max(0, now - obs.at) * 1000:.0f} ms"
            k = row_at(pred, now)
            prediction = pred.actions[k] if k is not None else None
            delay = f"GPU 前向 {pred.inference_ms:.1f} ms" if pred else "等待预测"
            src = state.source if state else "等待反馈"
            selected = robot.selected.rsplit("/", 1)[-1] or "未启动"
            chunked = robot.run_chunks and not robot.rtc_enabled
            if chunked:
                count = robot.execution_horizons.get(inputs.key, 12)
                rtc_status = f"连续执行 {count} 步，再读取新画面    30 Hz · RTC 关"
            elif robot.direct:
                rtc_status = "策略直出    30 Hz · 关节与夹爪目标进入右臂 mux"
            elif robot.rtc_enabled:
                rtc_status = (
                    f"RTC 开    已衔接 {robot.rtc.handoffs} 次    "
                    f"上次采用 {robot.rtc.prefix} 条指令"
                )
            elif trials is not None and trials.preset == "original":
                rtc_status = "原版执行    30 Hz · 逐拍限速 · RTC 关"
            else:
                rtc_status = "RTC 关    连续模式使用最新预测"
            units = "单位  关节 °   夹爪 mm"
            controls = "[空格] 暂停并保持当前位置    [Q] 退出"
            if not (robot.direct or chunked):
                units += (
                    f"      单条上限 {np.rad2deg(robot.travel.step[:7]).max():.1f}° / "
                    f"{robot.travel.step[7] * 1000:.0f} mm"
                )
            if (
                not robot.direct
                and trials is None
                and inputs.key
                and choices[inputs.key].rtc is not None
            ):
                controls = "[T] 切换 RTC    " + controls
            speed_label = "驱动启动速度" if robot.direct or chunked else "关节速度上限"
            run_keys = "[C] 接通右臂    [R] 连续    [N] 单步"
            if trials is not None:
                run_keys = "[C] 接通右臂    [R] 开始一次试验"
            if robot.reset_pose is not None:
                run_keys += "    [H] 可选复位"
            lines = [
                (f"OWAC  /  {task.upper()}       右从臂实机推理", True),
                (f"模式 {mode}     权重 {step}     {speed_label} {speed}%", True),
                ("", False),
                (detail, False),
                (f"右臂 {robot.can} · {src}    控制源 {selected}    {delay}", False),
                (f"已发策略指令 {robot.sent}    右臂 mux 输出 {robot.output_count}", False),
                (rtc_status, False),
                ("", False),
                (
                    "          "
                    + "".join(f"{name:>7}" for name in ("J1", "J2", "J3", "J4", "J5", "J6", "J7"))
                    + "    夹爪",
                    True,
                ),
            ]
            if len(weight_keys) > 1:
                shortcuts = "    ".join(
                    f"[{i}] {choices[key].label}" for i, key in enumerate(weight_keys[:2], 1)
                )
                lines.insert(2, (f"{shortcuts}    [M] 全部 {len(choices)} 套权重", False))
            if trials is not None:
                verdict_keys = (
                    "香蕉松开并留在篮内按 [S]，未完成按 [F]。两键都会暂停。"
                    if trials.preset == "placement"
                    else "香蕉夹起并离桌按 [S]，没抓起按 [F]。两键都会暂停。"
                )
                lines[2:2] = [
                    (trials.status(inputs.key, now), True),
                    (verdict_keys, True),
                ]
            if (
                state is not None
                and robot.reset_pose is not None
                and (not trials or not trials.active)
            ):
                delta = np.abs(state.values - robot.reset_pose)
                lines.insert(
                    4,
                    (
                        f"距示范起点  最大关节差 {np.rad2deg(delta[:7]).max():.1f}°"
                        f"   夹爪开度差 {delta[7] * 1000:.1f} mm    [H] 复位",
                        False,
                    ),
                )
            last_target = ("上次下发", robot.last_command)
            if robot.direct or chunked:
                issued = robot.last_policy_target
                last_target = (
                    "上次策略指令",
                    issued[1] if issued and issued[0] == inputs.key else None,
                )
            for label, values in (
                ("当前实测", state.values if state else None),
                ("复位目标", robot.reset_pose),
                ("模型预测", prediction),
                last_target,
            ):
                line = (
                    "   等待"
                    if values is None
                    else (
                        "".join(f"{v:7.1f}" for v in np.rad2deg(values[:7]))
                        + f"   {values[7] * 1000:5.1f}"
                    )
                )
                lines.append((label + "  " + line, False))
            lines.extend(
                [
                    (units, False),
                    ("", False),
                    (run_keys, True),
                    (controls, True),
                    (robot.info, False),
                    (trouble, False),
                ]
            )
            draw(screen, lines)
        time.sleep(0.004)


def readonly_check(inputs, robot, keys, count):
    import numpy as np

    from owac.runtime.bowl_postprocess import process_prediction
    from owac.runtime.fruit_live import FEEDBACK_SECONDS, command_row, row_at

    for key in keys:
        inputs.select(key)
        shadow = robot.rtc if keys[key].rtc is not None else None
        if shadow is not None:
            shadow.reset()
        ticks, prefixes = [], []

        def shadow_send(raw, ticks=ticks):
            wall = time.time_ns()
            feedback = inputs.at("ROS", wall) or inputs.at("CAN", wall)
            if feedback is None:
                raise RuntimeError("RTC 只读测试等待右臂新反馈")
            command = robot.travel.command(raw, feedback.values)
            now = time.monotonic()
            ticks.append(now)
            return now, command

        load_deadline = time.monotonic() + 120
        deadline = time.monotonic() + 25
        rows = []
        processor = robot.postprocessors.get(key) if not robot.direct else None
        processed = None
        if processor is not None:
            processor.reset()
        while len(rows) < count:
            with inputs.condition:
                timeout = (
                    max(0, shadow.next_tick - time.monotonic()) if shadow and shadow.plan else 0.1
                )
                inputs.condition.wait_for(
                    lambda last=rows[-1][0] if rows else -1: (
                        inputs.prediction is not None
                        and inputs.prediction.observation.capture > last
                    ),
                    timeout=timeout,
                )
                p = inputs.prediction
                latest = inputs.observation
                loading = inputs.loading == key
            now = time.monotonic()
            if loading:
                if now >= load_deadline:
                    raise RuntimeError(f"只读检查加载 {key} 超过 120 秒")
                deadline = now + 25
            if p and p.checkpoint == key and shadow is not None:
                if shadow.epoch is not None:
                    before = shadow.handoffs
                    shadow.step(p, now, shadow_send)
                    if shadow.handoffs != before:
                        prefixes.append(shadow.prefix)
                elif 0 <= now - p.observation.at <= FEEDBACK_SECONDS:
                    shadow.start(key, now)
            if p and p.checkpoint == key and (not rows or p.observation.capture > rows[-1][0]):
                obs = p.observation
                if processor is not None:
                    if p.scene is None:
                        raise ValueError("松爪后处理缺少当前双相机特征")
                    processed = process_prediction(
                        processor, p, keys[key].execution_horizon, 0, obs.feedback.values, now
                    )
                    processed = robot.physical_target(processed)
                deadline = now + 25
                k = row_at(p, now)
                if robot.direct and obs.feedback.source == "ROS":
                    try:
                        k = command_row(p, key, now, direct=True, latest_observation=latest)
                    except ValueError:
                        k = None
                rows.append(
                    (
                        obs.capture,
                        p.inference_ms,
                        k,
                        obs.feedback.source,
                        (obs.wall - obs.feedback.stamp) / 1e6,
                        [f.sequence for f in obs.frames],
                        p.actions.shape,
                        obs.feedback.values.tolist(),
                        p.actions[0].tolist(),
                    )
                )
            if time.monotonic() > deadline:
                with inputs.condition:
                    streams = (
                        len(inputs.receipts),
                        len(inputs.states["CAN"]),
                        len(inputs.states["ROS"]),
                    )
                    raise RuntimeError(
                        f"仅取得 {len(rows)} 次预测。"
                        f"近期双路相机/CAN/ROS 样本 {streams}。{inputs.events}"
                    )
        preview = {}
        if processed is not None:
            preview.update(
                postprocess_phase=processor.phase, postprocess_target_rad_m=processed.tolist()
            )
        if processed is not None and hasattr(processor, "diagnostics"):
            preview.update(processor.diagnostics())
        if robot.direct:
            k = rows[-1][2]
            preview = dict(
                preview_row=k, preview_target_rad_m=p.actions[k].tolist() if k is not None else None
            )
        print(
            json.dumps(
                dict(
                    policy=key,
                    step=keys[key].step,
                    direct=robot.direct,
                    pairs=len(rows),
                    source=rows[-1][3],
                    state_sources=dict(Counter(row[3] for row in rows)),
                    first_sequences=rows[0][5],
                    last_sequences=rows[-1][5],
                    shape=rows[-1][6],
                    median_inference_ms=np.median([r[1] for r in rows]),
                    first_inference_ms=rows[0][1],
                    first_in_time=rows[0][2] is not None,
                    median_state_age_ms=np.median([r[4] for r in rows]),
                    in_time=sum(r[2] is not None for r in rows),
                    measured_rad_m=rows[-1][7],
                    prediction_rad_m=rows[-1][8],
                    rtc_shadow_commands=len(ticks),
                    rtc_handoffs=shadow.handoffs if shadow else 0,
                    rtc_conditioned_handoffs=shadow.conditioned_handoffs if shadow else 0,
                    rtc_prefix_range=[min(prefixes), max(prefixes)] if prefixes else [],
                    rtc_median_interval_ms=np.median(np.diff(ticks)) * 1000
                    if len(ticks) > 1
                    else None,
                    policy_commands=robot.sent,
                    **preview,
                ),
                allow_nan=False,
            ),
            flush=True,
        )
        if not any(row[2] is not None for row in rows):
            raise RuntimeError("全部预测都已超出动作窗口")


def main():
    from owac.runtime.fruit_weights import FruitModels, catalog, default_policy, read_manifest

    root = Path(os.environ.get("OWAC_FRUIT_HOME", Path.home() / "owac/fruit-policy"))
    manifest = read_manifest(root)
    task = manifest.get("task", "fruit")
    parser = argparse.ArgumentParser(description=f"{task} 双相机与右从臂策略推理")
    parser.add_argument("--check", action="store_true", help="只读测试双路 RAW、状态及策略")
    parser.add_argument("--check-reset", action="store_true", help="只读检查 H 复位姿态并退出")
    parser.add_argument("--require-reset", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--list", action="store_true", help="列出权重并退出，无需连接设备")
    parser.add_argument("--policy", help="按权重 ID 进入预览或只读测试")
    parser.set_defaults(place_test=False, grasp_test=None)
    if task == "fruit":
        trial_group = parser.add_mutually_exclusive_group()
        trial_group.add_argument(
            "--place-test", action="store_true", help="记录搬运轨迹，并按香蕉是否留在篮内计分"
        )
        trial_group.add_argument(
            "--grasp-test",
            nargs="?",
            choices=("direct", "original"),
            const="direct",
            help="抓取计分。direct 用当前香蕉策略，original 用原版 2,500 步和加 RTC 前的发送方式",
        )
    parser.add_argument(
        "--direct", action="store_true", help="按 30 Hz 将回归策略输出直接送入原生从臂指令通道"
    )
    parser.add_argument("--frames", type=int, default=12, help="只读测试每个权重采样帧数")
    parser.add_argument("--speed", type=int, default=20, help="从臂驱动启动速度百分比，默认 20")
    args = parser.parse_args()
    choices = catalog(root, manifest)
    guard_profiles = manifest.get("pouring_grip_profiles", {})
    if guard_profiles and args.direct:
        parser.error("此倾倒候选使用反馈保护，请通过默认分段模式执行")
    if args.list:
        for key, choice in choices.items():
            print(f"{key:24} {choice.label}")
        return 0
    if not choices:
        parser.error(f"当前目录没有部署 {task} 权重")
    if args.grasp_test:
        if args.direct and args.grasp_test == "original":
            parser.error("original 抓取测试使用原版发送方式，请去掉 --direct")
        args.direct = args.grasp_test == "direct"
    args.policy = args.policy or default_policy(
        choices, check=args.check, grasp=args.grasp_test, place=args.place_test
    )
    if args.policy and args.policy not in choices:
        parser.error("未找到这个权重，可用 --list 查看")
    if not 1 <= args.speed <= 100 or args.frames < 1:
        parser.error("速度为 1–100，帧数必须为正数")
    from owac.runtime.fruit_live import Travel
    from owac.runtime.pouring_departure import PouringDeparture

    collection = Path(os.environ["OWAC_COLLECTION"])
    travel = Travel(collection / "agx_arm_urdf/nero/urdf", args.speed / 100)
    reset_path = root / "reset-pose.json"
    reset_error = None
    try:
        reset_pose = travel.load_reset_pose(reset_path)
    except (OSError, ValueError) as exc:
        if args.require_reset or args.check_reset:
            parser.error(f"无法载入 H 复位姿态 {reset_path}（{exc}）")
        reset_pose = None
        if not isinstance(exc, FileNotFoundError):
            reset_error = str(exc)
    departures = {
        key: PouringDeparture(json.loads((root / profile).read_text()), reset_pose)
        for key, profile in manifest.get("pouring_departures", {}).items()
    }
    if args.check_reset:
        print(
            json.dumps(
                dict(
                    task=task,
                    policy=args.policy,
                    path=str(reset_path),
                    reset_rad_m=reset_pose.tolist(),
                    **(
                        {"departure_profile": manifest["pouring_departures"][args.policy]}
                        if args.policy in departures
                        else {}
                    ),
                ),
                ensure_ascii=False,
            )
        )
        return 0
    if not args.check and not sys.stdin.isatty():
        parser.error("请在主机终端中运行，远程登录时加 ssh -t")
    # ROS Jazzy's Python and the OWAC virtualenv have the same system Python ABI.
    sys.path.insert(0, str(root / "deps"))
    sys.path.append("/usr/lib/python3/dist-packages")
    import rclpy
    import torch
    from pure_teleop.internal.can_setup import configured_bitrate, ensure_can_up
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.signals import SignalHandlerOptions

    from owac.raw.teleop_mcap import unpack_bg10
    from owac.runtime.bowl_postprocess import BowlPostprocessor
    from owac.runtime.fruit_live import STEPS, Inputs, passive_feedback
    from owac.runtime.fruit_robot import Equipment
    from owac.runtime.fruit_rtc import RegressionRTC
    from owac.runtime.pouring_grip import PouringGrip
    from owac.training.teleop_policy_eval import predict_actions, predict_commands

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    def finish_inputs():
        with inputs.condition:
            inputs.stopping = True
            inputs.condition.notify_all()
        worker.join(timeout=3)

    lockdir = Path.home() / ".cache/agilx-teleop"
    lockdir.mkdir(parents=True, exist_ok=True)
    can = os.environ["NERO_RIGHT_FOLLOWER_CAN"]
    try:
        with ExitStack() as stack:
            lock = stack.enter_context((lockdir / "right.lock").open("a"))
            attached = False
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                if not args.check:
                    raise RuntimeError("右侧录制台或推理台已打开，请先退出原终端中的程序") from None
                attached = True
            for sig in (signal.SIGTERM, signal.SIGHUP):
                stack.callback(signal.signal, sig, signal.signal(sig, interrupt))
            if not attached:
                ensure_can_up({"右从臂": can}, configured_bitrate())
            torch.set_num_threads(2)
            calibrated = any(choice.rtc is not None for choice in choices.values())
            adapters = {} if calibrated and not (args.direct or args.grasp_test) else None
            models = FruitModels(choices, adapters)
            selected = (
                [args.policy]
                if args.policy
                else [k for k in STEPS if k in choices] or list(choices)
            )
            print(f"正在加载 {choices[selected[0]].label}…", flush=True)
            models[selected[0]]
            rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
            stack.callback(rclpy.try_shutdown)
            node = rclpy.create_node(f"owac_fruit_console_{os.getpid()}")
            stack.callback(node.destroy_node)
            inputs = Inputs()
            if reset_error:
                inputs.events["复位配置"] = reset_error
            log_dir = root / "logs"
            command_log = None
            if manifest.get("record_policy_commands") and not args.check:
                log_dir.mkdir(parents=True, exist_ok=True)
                file = log_dir / f"commands-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}.jsonl"
                command_log = stack.enter_context(file.open("x", encoding="utf-8", buffering=1))
            robot = Equipment(
                node,
                inputs,
                collection,
                log_dir,
                can,
                travel,
                args.speed,
                reset_pose=reset_pose,
                rtc=RegressionRTC(adapters) if adapters is not None else None,
                direct=args.direct,
                trace_mux=bool(args.grasp_test or args.place_test or command_log),
                run_chunks=not (args.direct or args.grasp_test or args.check),
                execution_horizons={
                    key: choice.execution_horizon for key, choice in choices.items()
                },
                postprocessors={
                    key: BowlPostprocessor.load(choice.postprocess)
                    for key, choice in choices.items()
                    if choice.postprocess is not None
                },
                pouring_guards={
                    key: PouringGrip(json.loads((root / path).read_text()))
                    for key, path in guard_profiles.items()
                },
                pouring_departures=departures,
                physical_gripper_width=manifest.get("physical_gripper_width", False),
                command_log=command_log,
                command_frame_interval_s=manifest.get("command_frame_interval_s", 0),
            )
            stack.callback(robot.finish_frame_writes)
            executor = SingleThreadedExecutor()
            executor.add_node(node)
            listener = Thread(target=executor.spin, daemon=True)
            listener.start()
            stack.callback(listener.join, 2)
            stack.callback(executor.shutdown, timeout_sec=2)
            stop = Event()
            reader = Thread(
                target=passive_feedback, args=(inputs, can, stop, travel.upper[7]), daemon=True
            )
            reader.start()
            stack.callback(reader.join, 1)
            stack.callback(stop.set)
            predict = predict_commands if args.grasp_test == "original" else predict_actions
            worker = Thread(target=inputs.infer, args=(models, predict, unpack_bg10), daemon=True)
            worker.start()
            stack.callback(finish_inputs)
            stack.callback(robot.close)
            time.sleep(1.0)  # Allow local DDS endpoints to announce existing camera publishers.
            if not attached:
                robot.camera()
            if args.check:
                readonly_check(inputs, robot, {k: choices[k] for k in selected}, args.frames)
            else:
                if args.policy:
                    inputs.select(args.policy)
                    robot.info = "按 C 接通右臂，按 R 从当前位置开始"
                    if reset_pose is not None:
                        robot.info += "。需要回示范起点时按 H"
                trials = (
                    GraspTrials(root / "logs/grasp-tests", args.grasp_test, args.speed)
                    if args.grasp_test
                    else None
                )
                if args.place_test:
                    trials = GraspTrials(root / "logs/placement-tests", "placement", args.speed)
                try:
                    curses.wrapper(panel, inputs, robot, args.speed, choices, trials, task)
                finally:
                    if trials is not None:
                        trials.finish("unscored", time.monotonic(), robot.sent)
    except KeyboardInterrupt:
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"推理台  {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
