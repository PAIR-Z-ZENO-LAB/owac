"""Connect the fruit operator console to the installed NERO follower and camera bridge."""

import json
import os
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from functools import partial
from pathlib import Path

import numpy as np
from agx_arm_msgs.msg import AgxArmStatus, GripperStatus, OwacRawFrame
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from owac.raw.teleop_mcap import CAMERA_TOPICS, JOINTS, STATE, joint_positions
from owac.runtime.bowl_postprocess import process_prediction
from owac.runtime.fruit_live import FEEDBACK_SECONDS, HZ, Feedback, check_live_camera, command_row

NS = "/slave_right"
POLICY = NS + "/vla_joint_cmd"
MUX = NS + "/joint_cmd_mux"
LATCH = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
FOLLOWER = ("nero_ctrl_node", NS)
ROUTER = ("joint_cmd_mux_node", NS)


class Equipment:
    def __init__(
        self,
        node,
        inputs,
        collection,
        log_dir,
        can,
        travel,
        speed,
        *,
        reset_pose=None,
        rtc=None,
        direct=False,
        trace_mux=False,
        run_chunks=False,
        execution_horizons=None,
        postprocessors=None,
        pouring_guards=None,
        pouring_departures=None,
        physical_gripper_width=False,
        command_log=None,
        command_frame_interval_s=0,
    ):
        self.node, self.inputs, self.collection, self.log_dir = node, inputs, collection, log_dir
        self.can, self.travel, self.speed = can, travel, speed
        self.children = {}
        self.selected, self.grip, self.arm = "", None, None
        self.output_count = 0
        self.trace_mux = trace_mux
        self.pouring_departures = pouring_departures or {}
        self.physical_gripper_width = physical_gripper_width
        self.command_log = command_log
        if not np.isfinite(command_frame_interval_s) or command_frame_interval_s < 0:
            raise ValueError("现场画面记录间隔必须为非负有限值")
        self.command_frame_interval_s = command_frame_interval_s
        self.last_frame_sample = None
        self.pending_frame_write = None
        self.frame_writer = None
        self.frame_phase = None
        self.last_mux = None
        self.command_pub = self.select_pub = None
        self.engaging = 0
        self.selected_at = self.prime_at = 0.0
        self.wait_output = None
        self.starting_child = False
        self.armed = self.running = self.moved = False
        self.next_tick = 0.0
        self.sent = 0
        self.last_command = None
        self.last_policy_target = None
        self.last_model_target = None
        self.last_sample = None
        self.direct = direct
        self.run_chunks = run_chunks
        self.execution_horizons = execution_horizons or {}
        self.postprocessors = postprocessors or {}
        self.pouring_guards = pouring_guards or {}
        self.chunk = None
        self.chunk_row = 0
        self.rtc = None if direct else rtc
        self.rtc_enabled = self.rtc is not None and not run_chunks
        self.reset_pose = reset_pose
        self.reset_deadline = None
        self.reset_final_sent_ns = None
        self.policy_after = 0.0
        self.info = "按 C 接通后可复位，按 1 或 2 预览策略"
        self.graph_checked = 0.0
        self.graph_problem = ""
        for i, topic in enumerate(CAMERA_TOPICS):
            node.create_subscription(
                OwacRawFrame, topic, partial(self.image, i), qos_profile_sensor_data
            )
        for typ, topic, cb, qos in (
            (JointState, STATE, self.state, qos_profile_sensor_data),
            (GripperStatus, NS + "/gripper_status", self.gripper_status, qos_profile_sensor_data),
            (AgxArmStatus, NS + "/arm_status", self.arm_status, qos_profile_sensor_data),
            (String, MUX + "/selected", self.selection, LATCH),
            (JointState, MUX + "/output", self.output, qos_profile_sensor_data),
        ):
            node.create_subscription(typ, topic, cb, qos)

    def image(self, view, msg):
        try:
            self.inputs.frame(view, msg)
        except ValueError as exc:
            self.inputs.events["相机"] = str(exc)
        else:
            self.inputs.events.pop("相机", None)

    def state(self, msg):
        try:
            values = joint_positions(msg)
        except (ValueError, KeyError):
            return
        stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        self.inputs.feedback(Feedback(stamp, time.time_ns(), values, "ROS"))

    def gripper_status(self, msg):
        self.grip = time.monotonic(), msg

    def arm_status(self, msg):
        self.arm = time.monotonic(), msg

    def selection(self, msg):
        self.selected = msg.data

    def output(self, msg):
        if len(msg.position) == len(JOINTS):
            self.output_count += 1
            if self.trace_mux:
                try:
                    positions = joint_positions(msg)
                except (KeyError, ValueError):
                    return
                stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
                self.last_mux = stamp, positions

    def endpoints(self, topic, *, publishers=True):
        lookup = (
            self.node.get_publishers_info_by_topic
            if publishers
            else self.node.get_subscriptions_info_by_topic
        )
        return [(p.node_name, p.node_namespace) for p in lookup(topic)]

    def start_child(self, name, arguments):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}.log"
        with path.open("w") as log:
            proc = subprocess.Popen(
                arguments,
                start_new_session=True,
                cwd=self.collection,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        self.children[name] = (proc, path)

    def camera(self):
        publishers = [self.endpoints(topic) for topic in CAMERA_TOPICS]
        if any(publishers):
            if not all(p == [("owac_raw_camera", "/")] for p in publishers):
                raise RuntimeError("U2/U3 相机发布者冲突，请关闭重复采集进程")
            return
        import yaml

        board = yaml.safe_load((self.collection / "config/recording-right-owac.yaml").read_text())
        spec = board["sources"]["owac"]
        self.start_child(
            "camera",
            [
                os.environ["PURE_TELEOP_ROS2_PYTHON"],
                "scripts/owac_raw_camera_ros2.py",
                "--host",
                str(spec["host"]),
                "--port",
                str(spec["port"]),
                "--fps",
                str(spec["fps"]),
            ],
        )

    def connect(self):
        if self.engaging or self.armed:
            return
        wall = time.time_ns()
        if not (self.inputs.at("ROS", wall) or self.inputs.at("CAN", wall)):
            raise RuntimeError("等待右从臂实时反馈，请检查供电和 CAN 连接后再按 C")
        if self.command_pub is None:
            self.command_pub = self.node.create_publisher(JointState, POLICY, 1)
        identity = (self.node.get_name(), self.node.get_namespace())
        if any(p != identity for p in self.endpoints(POLICY)):
            raise RuntimeError("另一策略正在发布右臂指令，请先退出它")
        from owac.runtime.nero_controllers import local_controllers

        controllers = local_controllers()
        if any(((port == self.can) != (ns == NS)) for ns, port in controllers):
            raise RuntimeError("已有驱动的 CAN 与右从臂配置冲突，请检查设备映射")
        published = self.endpoints(STATE)
        if published and (published != [FOLLOWER] or (NS, self.can) not in controllers):
            raise RuntimeError("右从臂反馈发布者与本机 CAN 驱动不一致")
        if not published and (NS, self.can) not in controllers:
            if self.endpoints(MUX + "/selected"):
                raise RuntimeError("发现已有右臂指令节点，请先检查原驱动终端")
            if "follower" in self.children and self.children["follower"][0].poll() is None:
                raise RuntimeError("右从臂驱动仍在启动，请稍候")
            command = (
                "source scripts/use_teleop.sh && teleop_preset_nero_gripper_v112_no_camera "
                "&& export ENABLE_CMD_MUX=true CMD_MUX_DEFAULT_SOURCE=policy "
                '&& exec ./scripts/run_teleop.sh follower "speed_percent:=$1"'
            )
            self.start_child("follower", ["bash", "-c", command, "fruit", str(self.speed)])
            self.starting_child = True
        self.engaging = time.monotonic()
        self.selected_at = self.prime_at = 0.0
        self.wait_output = None
        self.info = "正在连接右从臂驱动与策略通道"

    def graph_error(self, now):
        if now - self.graph_checked >= FEEDBACK_SECONDS:
            identity = (self.node.get_name(), self.node.get_namespace())
            ok = (
                self.endpoints(STATE) == [FOLLOWER]
                and self.endpoints(POLICY) == [identity]
                and self.endpoints(POLICY, publishers=False).count(ROUTER) == 1
                and self.endpoints(MUX + "/selected") == [ROUTER]
                and self.endpoints(MUX + "/output") == [ROUTER]
                and FOLLOWER in self.endpoints(MUX + "/output", publishers=False)
            )
            self.graph_problem = "" if ok else "右从臂驱动、策略发布者或 mux 通道未就绪"
            self.graph_checked = now
        return self.graph_problem

    def health(self, now):
        # DDS discovery may yield to the telemetry callbacks. Check ages after it returns.
        graph_problem = self.graph_error(now)
        if self.inputs.at("ROS", time.time_ns()) is None:
            return "等待右从臂 ROS 关节反馈"
        now = time.monotonic()
        if not self.arm or now - self.arm[0] > FEEDBACK_SECONDS:
            return "等待右从臂硬件状态"
        arm = self.arm[1]
        if arm.arm_status or arm.err_status:
            return f"右从臂状态 {arm.arm_status}，错误码 {arm.err_status}"
        if not self.grip or now - self.grip[0] > FEEDBACK_SECONDS:
            return "等待夹爪硬件反馈"
        gripper = self.grip[1]
        from piper_nero_teleop.gripper_width import formal_width_from_status

        faults = (
            "voltage_too_low",
            "motor_overheating",
            "driver_overcurrent",
            "driver_overheating",
            "sensor_status",
            "driver_error_status",
        )
        if (
            not gripper.driver_enable_status
            or any(getattr(gripper, k) for k in faults)
            or formal_width_from_status(gripper, self.travel.upper[7]) is None
        ):
            return "夹爪尚未就绪，请检查使能、状态和宽度"
        with self.inputs.condition:
            motors = list(self.inputs.motors.values())
        if len(motors) != 7 or any(now - t > 0.5 or (flags & 0x7F) != 0x40 for t, flags in motors):
            return "等待七个关节报告已使能且无驱动故障"
        return graph_problem

    def publish(self, targets):
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = list(JOINTS)
        msg.position = list(map(float, targets))
        self.command_pub.publish(msg)

    def physical_target(self, target):
        if self.physical_gripper_width:
            # The NERO width driver takes abs(raw). A negative predicted aperture means closed.
            target = target.copy()
            target[7] = np.clip(target[7], self.travel.lower[7], self.travel.upper[7])
        return target

    def save_phase_frame(self, processor, observation):
        if self.frame_phase == processor.phase:
            return None
        self.frame_phase = processor.phase
        return self.save_observation_frame(processor.phase, observation)

    def trace_observation(self, observation, *, pause=False):
        if self.command_log is None or not self.command_frame_interval_s or observation is None:
            return None
        if self.last_frame_sample is not None:
            capture, at, saved = self.last_frame_sample
            if observation.capture == capture:
                return saved if pause else None
            if not pause and observation.at - at < self.command_frame_interval_s:
                return None
        if self.pending_frame_write is not None and not self.pending_frame_write.done():
            self.inputs.events["现场画面"] = "写盘忙，已跳过本次画面；指令日志继续"
            return None
        saved = self.save_observation_frame("pause" if pause else "sample", observation)
        self.last_frame_sample = observation.capture, observation.at, saved
        return saved

    def save_observation_frame(self, label, observation):
        raw = [np.frombuffer(bytes(f.data), np.uint8) for f in observation.frames]
        folder = Path(self.command_log.name).with_suffix(".frames")
        file = folder / f"{observation.capture}-{label}.npz"

        def write():
            try:
                folder.mkdir(exist_ok=True)
                np.savez(
                    file,
                    u2_bg10=raw[0],
                    u3_bg10=raw[1],
                    state=observation.feedback.values,
                    width=observation.frames[0].width,
                    height=observation.frames[0].height,
                    stride=observation.frames[0].stride,
                    wall_ns=observation.wall,
                )
            except OSError as exc:
                self.inputs.events["阶段画面"] = str(exc)

        if self.frame_writer is None:
            self.frame_writer = ThreadPoolExecutor(max_workers=1)
        self.pending_frame_write = self.frame_writer.submit(write)
        return str(file.relative_to(folder.parent))

    def finish_frame_writes(self):
        if self.frame_writer is not None:
            self.frame_writer.shutdown()

    def publish_policy(self, target, prediction, row, *, processor=None):
        target = self.physical_target(target)
        self.publish(target)
        if self.command_log is not None:
            wall = time.time_ns()
            feedback = self.inputs.at("ROS", wall)
            mux = self.last_mux
            record = dict(
                time_ns=wall,
                policy=prediction.checkpoint,
                capture=prediction.observation.capture,
                row=row,
                observation_rad_m=prediction.observation.feedback.values.tolist(),
                model_rad_m=prediction.actions[row].tolist(),
                issued_rad_m=target.tolist(),
                latest_mux_ns=mux[0] if mux else None,
                latest_mux_rad_m=mux[1].tolist() if mux else None,
                feedback_ns=feedback.stamp if feedback else None,
                feedback_rad_m=feedback.values.tolist() if feedback else None,
            )
            if departure := self.pouring_departures.get(prediction.checkpoint):
                record["pouring_departure"] = departure.phase
            if self.command_frame_interval_s:
                observation = prediction.observation
                record["camera_wall_ns"] = observation.wall
                record["camera_sequences"] = [frame.sequence for frame in observation.frames]
                sampled = self.trace_observation(observation)
                if sampled:
                    record["sample_frame"] = sampled
            if processor is not None:
                record.update(postprocess_phase=processor.phase, release_score=processor.score)
                if hasattr(processor, "diagnostics"):
                    record.update(processor.diagnostics())
                    phase_frame = self.save_phase_frame(processor, prediction.observation)
                    if phase_frame:
                        record["phase_frame"] = phase_frame
                    if self.grip is not None:
                        record["gripper_force"] = float(self.grip[1].force)
            if guard := self.pouring_guards.get(prediction.checkpoint):
                record.update(guard.diagnostics())
            try:
                self.command_log.write(json.dumps(record) + "\n")
            except OSError as exc:
                self.inputs.events["指令日志"] = str(exc)
        return target

    def pause(self, reason, *, decision=None):
        event = None
        if self.log_dir is not None and (self.armed or self.engaging or self.running):
            with self.inputs.condition:
                obs = self.inputs.observation
                now = time.monotonic()
                at, pred = decision if decision is not None else (now, self.inputs.prediction)
                event = {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S%z"),
                    "policy": self.inputs.key,
                    "direct": self.direct,
                    "rtc_enabled": self.rtc_enabled,
                    "running": self.running,
                    "selected": self.selected,
                    "sent": self.sent,
                    "latest_frame_ms": round((now - obs.at) * 1000, 1) if obs else None,
                    "prediction_frame_ms": round((at - pred.observation.at) * 1000, 1)
                    if pred
                    else None,
                    "forward_ms": round(pred.inference_ms, 1) if pred else None,
                    "prediction_state": pred.observation.feedback.source if pred else None,
                    "input_errors": dict(self.inputs.events),
                }
                if (self.direct or self.run_chunks) and self.last_policy_target is not None:
                    last_key, target = self.last_policy_target
                    if last_key == self.inputs.key:
                        event["last_policy_target_rad_m"] = target.tolist()
        if self.rtc is not None:
            self.rtc.reset()
        departure = self.pouring_departures.get(self.inputs.key)
        if departure is not None:
            departure.suspend()
            if event is not None:
                event["pouring_departure"] = departure.phase
        for processor in self.postprocessors.values():
            processor.suspend()
        guard = self.pouring_guards.get(self.inputs.key)
        if guard:
            guard.suspend()
            if event is not None:
                event.update(guard.diagnostics())
                event["pause_time_ns"] = time.time_ns()
        self.chunk = None
        if self.reset_deadline is not None:
            self.finish_reset(time.monotonic())
        if self.moved and self.selected == POLICY:
            now = time.time_ns()
            fb = self.inputs.at("ROS", now) or self.inputs.at("CAN", now)
            if fb:
                target = fb.values.copy()
                if guard and guard.carrying and self.last_command is not None:
                    target[7] = self.last_command[7]
                self.publish(target)
                self.last_command = target
            else:
                reason += "。反馈中断，驱动保留上一目标"
        if self.starting_child:
            self.starting_child = False
            self.stop_child("follower")
        self.clear_selector()
        self.engaging = 0
        self.running = self.moved = False
        self.info = reason
        if event is not None:
            event["reason"] = reason
            if self.command_frame_interval_s:
                event["command_file"] = (
                    Path(self.command_log.name).name if self.command_log else None
                )
                event["pause_frame"] = self.trace_observation(obs, pause=True)
                self.last_frame_sample = None
            try:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                with (self.log_dir / "console-pauses.jsonl").open("a", encoding="utf-8") as log:
                    log.write(json.dumps(event, ensure_ascii=False) + "\n")
            except OSError as exc:
                self.inputs.events["暂停日志"] = str(exc)

    def toggle_rtc(self):
        if self.rtc is None or self.inputs.key not in self.rtc.adapters:
            self.info = "当前策略使用原生动作块"
            return
        self.pause("已暂停")
        self.rtc_enabled = not self.rtc_enabled
        mode = "RTC" if self.rtc_enabled else "直接回归"
        self.info = f"已切换到{mode}，按 R 开始连续执行"

    def start_reset(self, now):
        if self.reset_deadline is not None:
            return
        if not self.armed:
            self.info = "先按 C 接通右从臂，再按 H 回到起点"
            return
        self.pause("准备复位")
        if self.reset_pose is None:
            self.info = "复位姿态不可用，请检查 reset-pose.json"
            return
        issue = self.health(now)
        feedback = self.inputs.at("ROS", time.time_ns())
        if issue or self.selected != POLICY or feedback is None:
            self.info = issue or "复位需要右从臂的控制权和新鲜反馈"
            return
        planned = self.travel.reset_ticks(self.reset_pose, feedback.values) / HZ
        self.frame_phase = None
        for processor in self.postprocessors.values():
            processor.reset()
        self.reset_deadline = now + planned + 15.0  # allow time for physical tracking to settle
        self.reset_final_sent_ns = None
        self.last_sample = None
        self.info = "正在回到示范起点，空格可中止"

    def finish_reset(self, now):
        self.reset_deadline = None
        self.reset_final_sent_ns = None
        self.policy_after = now
        self.inputs.begin_motion(now)
        self.running = False

    def reset_tick(self, now):
        fb = self.inputs.at("ROS", time.time_ns())
        if fb is None:
            self.pause("复位已中止，右从臂反馈中断")
            return
        error = np.abs(self.reset_pose - fb.values)
        joint, width = float(np.max(error[:7])), float(error[7])
        residual = f"最大关节差 {np.rad2deg(joint):.2f}°，夹爪差 {width * 1000:.1f} mm"
        # The driver's servo resend deadband is not its physical positioning accuracy.
        # A post-command reading inside the last reset waypoint is enough to start a trial.
        if (
            self.reset_final_sent_ns is not None
            and fb.stamp > self.reset_final_sent_ns
            and self.travel.reset_ticks(self.reset_pose, fb.values) == 1
        ):
            self.finish_reset(now)
            for departure in self.pouring_departures.values():
                departure.reset()
            for guard in self.pouring_guards.values():
                guard.reset()
            self.info = f"复位完成，等待手动执行。{residual}"
            return
        if now >= self.reset_deadline:
            self.pause(f"复位超时，已暂停。{residual}")
            return
        try:
            target, final = self.travel.reset_command(self.reset_pose, fb.values)
        except ValueError as exc:
            self.pause(str(exc))
            return
        self.publish(target)
        if final:
            if self.reset_final_sent_ns is None:
                self.reset_final_sent_ns = time.time_ns()
        else:
            self.reset_final_sent_ns = None
        self.last_command, self.moved = target, True
        self.next_tick = now + 1 / HZ
        self.info = f"复位中。{residual}"

    def tick(self, now):
        if self.engaging:
            child = self.children.get("follower")
            if child and child[0].poll() is not None:
                self.pause(f"右从臂驱动已退出，日志 {child[1]}")
                return
            problem = self.health(now)
            if now - self.engaging > 45:
                self.pause(problem or "切换策略控制源超时")
            elif problem:
                self.info = problem
            elif self.selected != POLICY:
                # The mux may still have a command cached for 0.5 s. Prime with the
                # measured pose before selecting the port; never replay that cached target.
                self.wait_output = None
                fb = self.inputs.at("ROS", time.time_ns())
                if fb is not None and now >= self.next_tick:
                    self.publish(fb.values)
                    self.next_tick = now + 1 / HZ
                    self.prime_at = self.prime_at or now
                    if now - self.prime_at >= 0.5 and now - self.selected_at >= 0.5:
                        if self.select_pub is None:
                            self.select_pub = self.node.create_publisher(
                                String, MUX + "/select", LATCH
                            )
                        if self.select_pub.get_subscription_count():
                            self.select_pub.publish(String(data="policy"))
                            self.selected_at = now
            elif self.wait_output is None or self.output_count <= self.wait_output:
                fb = self.inputs.at("ROS", time.time_ns())
                if fb is not None and now >= self.next_tick:
                    self.wait_output = self.output_count
                    self.publish(fb.values)
                    self.next_tick = now + 1 / HZ
                    self.info = "等待 mux 回传当前姿态"
            else:
                self.engaging = 0
                self.starting_child = False
                self.clear_selector()
                self.armed = True
                self.info = "已接通右从臂，按 H 复位或选择动作"
        elif self.armed:
            problem = self.health(now)
            if problem or self.selected != POLICY:
                self.pause(problem or f"右臂控制源已切换为 {self.selected or '未选择'}")
                self.armed = False
            elif self.reset_deadline is not None and now >= self.next_tick:
                self.reset_tick(now)
            elif self.running and now >= self.next_tick:
                self.step(now)

    def step(self, now, *, continuous=None):
        if self.reset_deadline is not None:
            self.info = "复位中，等待到位或按空格中止"
            return
        if not self.armed:
            self.info = "请先按 C 接通右从臂"
            return
        if self.inputs.key is None:
            self.info = "请先按 1 或 2 选择权重"
            return
        issue = self.health(now)
        if issue or self.selected != POLICY:
            self.pause(issue or f"右臂控制源已切换为 {self.selected or '未选择'}")
            self.armed = False
            return
        with self.inputs.condition:
            fb = self.inputs.at("ROS", time.time_ns())
            pred = self.inputs.prediction
            latest = self.inputs.observation
            key = self.inputs.key
            # Compare this snapshot to its consumption time, after the UI and ROS checks.
            now = time.monotonic()
        if fb is None:
            self.pause("执行已暂停，右臂 ROS 关节反馈已过期", decision=(now, pred))
            return
        try:
            if self.run_chunks and not self.rtc_enabled:
                self.step_chunk(key, pred, latest, now, continuous, fb.values)
                return
            if key in self.pouring_departures:
                raise ValueError("此倾倒策略需要分段起步，请使用默认模式")
            k = command_row(pred, key, now, direct=not self.rtc_enabled, latest_observation=latest)
        except ValueError as exc:
            self.pause(f"执行已暂停，{exc}", decision=(now, pred))
            return
        if pred.observation.at <= self.policy_after:
            self.pause("等待复位后的新画面，随后可按 N 或 R", decision=(now, pred))
            return
        if continuous is not None:
            self.running = continuous
            if not continuous and self.rtc is not None:
                self.rtc.reset()
        use_rtc = self.running and self.rtc_enabled
        if use_rtc and self.rtc.epoch is None:
            self.rtc.start(self.inputs.key, now)
            self.info = "RTC 等待本次启动后的新画面"
        if now < self.next_tick:
            return

        def send(raw):
            target = raw if self.direct else self.travel.command(raw, fb.values)
            target = self.publish_policy(target, pred, k)
            return time.monotonic(), target

        if use_rtc:
            try:
                action = self.rtc.step(pred, time.monotonic(), send)
            except (RuntimeError, np.linalg.LinAlgError) as exc:
                self.pause(str(exc), decision=(now, pred))
                return
            self.next_tick = self.rtc.next_tick
            if action is None:
                return
            capture, k, target = action
        else:
            self.next_tick = now + 1 / HZ
            capture = pred.observation.capture
            if (capture, k) == self.last_sample:
                return
            _, target = send(pred.actions[k])
        self.sent += 1
        if self.direct or self.trace_mux or self.run_chunks:
            self.last_policy_target = key, target.copy()
        if self.trace_mux:
            self.last_model_target = pred.actions[k].copy()
        self.last_command, self.last_sample, self.moved = target, (capture, k), True
        self.info = f"已发第 {self.sent} 条策略指令，使用窗口第 {k + 1} 项"

    def step_chunk(self, key, prediction, latest, now, continuous, measured):
        """Execute the selected prefix in order, then wait for a fresh observation."""
        check_live_camera(latest, now)
        if continuous is not None:
            if not continuous or not self.running:
                self.chunk = None
            if continuous and not self.running:
                self.policy_after = now
                self.inputs.begin_motion(now)
            self.running = continuous
        if now < self.next_tick:
            return
        if self.chunk is not None and now - self.next_tick > FEEDBACK_SECONDS:
            self.chunk = None
            self.policy_after = now
            if processor := self.postprocessors.get(key):
                processor.suspend()
            if guard := self.pouring_guards.get(key):
                guard.suspend()
        period = 1 / HZ
        self.next_tick = (self.next_tick if now - self.next_tick < period else now) + period
        if self.chunk is None:
            if prediction is None or prediction.observation.at <= self.policy_after:
                self.info = "等待本次动作后的新画面和策略预测"
                return
            command_row(prediction, key, now, direct=True, latest_observation=latest)
            self.chunk = prediction
            self.chunk_row = 0
        plan, row = self.chunk, self.chunk_row
        count = self.execution_horizons.get(key, len(plan.actions))
        processor = self.postprocessors.get(key)
        target = plan.actions[row]
        departure = self.pouring_departures.get(key)
        if departure is not None:
            target = departure.command(
                plan,
                row,
                count,
                measured,
                self.travel,
                now,
                rate_limit=key not in self.pouring_guards,
            )
        if processor is not None:
            target = process_prediction(processor, plan, count, row, measured, now)
        guard = self.pouring_guards.get(key)
        if guard is not None:
            target = guard.command(target, measured, self.travel, now)
        target = self.publish_policy(target, plan, row, processor=processor)
        self.sent += 1
        self.last_command = target.copy()
        self.last_policy_target = key, self.last_command
        self.last_model_target = plan.actions[row].copy()
        self.last_sample = (plan.observation.capture, row)
        self.moved = True
        self.info = f"已发第 {self.sent} 条策略指令，本段 {row + 1}/{count}"
        if departure is not None:
            self.info += f" · {departure.phase}"
        if guard is not None:
            self.info += f" · {guard.phase}"
        if processor is not None and hasattr(processor, "diagnostics"):
            self.info += f" · {processor.phase}"
            if processor.phase == "finished":
                evidence = (
                    "顶视画面识别到黑碗位于叠放处"
                    if processor.stack_verdict == "black"
                    else "叠放处的黑碗待现场复核"
                )
                self.pause(f"右臂已退离，{evidence}。按 H 可复位")
                return
        self.chunk_row += 1
        if not self.running or self.chunk_row == count:
            self.chunk = None
            self.policy_after = time.monotonic()

    def clear_selector(self):
        if self.select_pub is not None:
            self.node.destroy_publisher(self.select_pub)
            self.select_pub = None

    def stop_child(self, name):
        proc, _ = self.children.pop(name)
        for sig, timeout in ((signal.SIGINT, 8), (signal.SIGTERM, 4), (signal.SIGKILL, None)):
            if proc.poll() is not None:
                return
            with suppress(ProcessLookupError):
                os.killpg(proc.pid, sig)
            try:
                proc.wait(timeout=timeout)
                return
            except subprocess.TimeoutExpired:
                continue

    def close(self):
        try:
            self.pause("退出")
        finally:
            for name in reversed(list(self.children)):
                self.stop_child(name)
