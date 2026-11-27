"""Stereo capture/state pairing, action timing and physical command limits for live fruit trials."""

import json
import math
import socket
import struct
import time
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass
from threading import Condition

import numpy as np

from owac.raw.teleop_mcap import JOINTS

FEEDBACK_SECONDS = 0.2  # robot_feedback_timeout_sec in the NERO launch configuration
HZ = 30
STEPS = {"best": 2500, "final": 10000}


@dataclass
class Feedback:
    stamp: int  # host Unix ns, same time base as the recorded state header
    arrived: int
    values: np.ndarray
    source: str


@dataclass
class Observation:
    capture: int
    at: float  # later of the two camera host receive times, monotonic seconds
    wall: int
    frames: tuple
    feedback: Feedback


@dataclass
class Prediction:
    checkpoint: str
    observation: Observation
    actions: np.ndarray
    inference_ms: float
    scene: np.ndarray | None = None


class Inputs:
    def __init__(self):
        self.condition = Condition()
        self.states = {source: deque(maxlen=512) for source in ("ROS", "CAN")}
        self.views = {}
        self.observation = None
        # RawVLA uses six consecutive-frame bursts with recurrent anchors five frames apart.
        self.history = deque(maxlen=31)
        self.receipts = deque(maxlen=32)
        self.previous_board = [None, None]
        self.events = {}
        self.motors = {}
        self.key = None
        self.loading = None
        self.generation = 0
        self.prediction = None
        self.motion_started_at = 0.0
        self.stopping = False

    def feedback(self, sample):
        with self.condition:
            self.states[sample.source].append(sample)

    def at(self, source, wall):
        with self.condition:
            eligible = (
                sample
                for sample in self.states[source]
                if 0 <= wall - sample.stamp <= FEEDBACK_SECONDS * 1e9 and sample.arrived <= wall
            )
            return max(eligible, key=lambda s: (s.stamp, s.arrived), default=None)

    def frame(self, view, msg):
        if (msg.camera_id, msg.fourcc, msg.width, msg.height, msg.stride) != (
            view,
            "BG10",
            640,
            480,
            1024,
        ) or len(msg.data) != msg.height * msg.stride:
            raise ValueError("相机格式应为 U2/U3 的 640×480 BG10")
        with self.condition:
            request = msg.request_sent_monotonic_ns
            pair = self.views.setdefault(request, {})
            pair[view] = msg
            if len(self.views) > 16:
                self.views.pop(min(self.views))
            if len(pair) != 2:
                return
            self.views.pop(request, None)
            frames = (pair[0], pair[1])
            if frames[0].capture_done_monotonic_ns != frames[1].capture_done_monotonic_ns:
                raise ValueError("同一 capture 的两路完成时间不一致")
            if self.observation and request <= self.observation.capture:
                return
            board = [f.board_timestamp_us for f in frames]
            if any(a == b for a, b in zip(board, self.previous_board, strict=True)):
                return
            self.previous_board = board
            wall = max(f.header.stamp.sec * 10**9 + f.header.stamp.nanosec for f in frames)
            at = max(f.received_monotonic_ns for f in frames) / 1e9
            self.receipts.append(at)
            state = self.at("ROS", wall) or self.at("CAN", wall)
            if state is not None:
                if self.observation and at - self.observation.at > FEEDBACK_SECONDS:
                    self.history.clear()
                self.observation = Observation(request, at, wall, frames, state)
                self.history.append(self.observation)
                self.condition.notify_all()

    def select(self, key):
        with self.condition:
            self.key = key
            self.generation += 1
            self.prediction = None
            self.loading = None
            self.history.clear()
            self.motion_started_at = 0.0
            if self.observation is not None:
                self.history.append(self.observation)
            self.events.pop("模型", None)
            self.condition.notify_all()

    def begin_motion(self, now):
        with self.condition:
            self.motion_started_at = now

    def history_for(self, model, observation):
        if getattr(model, "cfg", {}).get("model", {}).get("state_history_lags"):
            return tuple(s for s in self.history if s.at >= self.motion_started_at) or (
                observation,
            )
        return tuple(self.history)

    def infer(self, models, predict, unpack):
        seen = None
        while True:
            with self.condition:
                self.condition.wait_for(
                    lambda seen=seen: (
                        self.stopping
                        or (
                            self.key
                            and self.observation
                            and (self.generation, self.observation.capture) != seen
                        )
                    )
                )
                if self.stopping:
                    return
                key, generation = self.key, self.generation
                if key not in models:
                    self.loading = key
                    self.events["模型"] = f"正在加载 {key}…"
            try:
                model = models[key]
            except Exception as exc:
                with self.condition:
                    if self.generation == generation:
                        self.loading = None
                        self.events["模型"] = str(exc)
                        self.condition.notify_all()
                        self.condition.wait_for(
                            lambda generation=generation: (
                                self.stopping or self.generation != generation
                            )
                        )
                continue
            try:
                # A weight change can take seconds. Use the latest pair after loading.
                with self.condition:
                    if self.generation != generation:
                        continue
                    self.loading = None
                    obs = self.observation
                    history = self.history_for(model, obs)
                    seen = generation, obs.capture
                start = time.monotonic()
                scene = None
                if getattr(model, "return_live_scene", False):
                    actions, scene = model.predict_live(history, unpack, return_scene=True)
                    if not np.isfinite(scene).all():
                        raise ValueError("放碗判断收到无效的视觉特征")
                elif hasattr(model, "predict_live"):
                    actions = model.predict_live(history, unpack)
                else:
                    raw = np.stack(
                        [unpack(f.data, f.width, f.height, f.stride) for f in obs.frames]
                    )
                    actions = predict(model, raw, obs.feedback.values)[0]
                elapsed = (time.monotonic() - start) * 1000
                if actions.shape != (12, 8) or not np.isfinite(actions).all():
                    raise ValueError("策略输出不是 12×8 的有限数值")
                result = Prediction(key, obs, actions, elapsed, scene)
                with self.condition:
                    if self.generation == generation:
                        self.prediction = result
                        self.events.pop("模型", None)
                        self.condition.notify_all()
            except Exception as exc:
                with self.condition:
                    if self.generation == generation:
                        self.prediction = None
                        self.events["模型"] = str(exc)
                        self.condition.notify_all()
            finally:
                del model


def row_at(prediction, now):
    if prediction is None:
        return None
    age = now - prediction.observation.at
    if age < 0:
        return None
    row = int(age * HZ + 0.5)
    return row if row < len(prediction.actions) else None


def check_live_camera(observation, now):
    if observation is None:
        raise ValueError("等待双路相机和新鲜关节反馈")
    age = now - observation.at
    if age < 0 or now > observation.at + FEEDBACK_SECONDS:
        raise ValueError(
            f"最新配对画面距今 {age * 1000:.1f} ms，上限 {FEEDBACK_SECONDS * 1000:g} ms"
        )


def command_row(prediction, key, now, *, direct=False, latest_observation=None):
    """Pick an executable row from a prediction at the time it is consumed."""
    if prediction is None:
        raise ValueError("尚未收到当前策略的预测")
    if prediction.checkpoint != key:
        raise ValueError(f"预测来自 {prediction.checkpoint}，当前权重为 {key}")
    obs = prediction.observation
    if obs.feedback.source != "ROS":
        raise ValueError(f"画面配对的是 {obs.feedback.source} 反馈，执行需要右臂 ROS 反馈")
    age = now - obs.at
    if age < 0:
        raise ValueError(f"相机时间比主机当前时间晚 {-age * 1000:.1f} ms")
    max_age = FEEDBACK_SECONDS
    if direct:
        # Check the live camera separately from a GPU result that may arrive several rows late.
        latest = latest_observation if latest_observation is not None else obs
        check_live_camera(latest, now)
        max_age = (len(prediction.actions) - 1) / HZ
    if now > obs.at + max_age:
        raise ValueError(f"预测所用画面距今 {age * 1000:.1f} ms，上限 {max_age * 1000:g} ms")
    row = row_at(prediction, now)
    if row is None:
        raise ValueError("本次预测的动作序列已用完")
    return row


class Travel:
    def __init__(self, urdf_folder, fraction):
        robot = ET.parse(urdf_folder / "nero_description.urdf").getroot()
        grip = ET.parse(urdf_folder / "nero_with_gripper_description.xacro").getroot()
        limits = [robot.find(f".//joint[@name='{n}']/limit").attrib for n in JOINTS[:7]]
        fingers = [grip.find(f".//joint[@name='gripper_joint{i}']/limit").attrib for i in (1, 2)]
        width = sum(float(f["upper"]) - float(f["lower"]) for f in fingers)
        self.lower = np.array([float(v["lower"]) for v in limits] + [0.0])
        self.upper = np.array([float(v["upper"]) for v in limits] + [width])
        velocities = [float(v["velocity"]) for v in limits]
        velocities.append(sum(float(f["velocity"]) for f in fingers))
        # Per-command increments from this host's stereo-act conservative deployment preset.
        # Keep the actual NERO finger travel, which spans 0..100 mm on this installation.
        jog_step = np.array([0.03] * 7 + [0.003])
        self.step = np.minimum(np.array(velocities) * fraction / HZ, jog_step)

    def command(self, target, current):
        bounded = np.clip(target, self.lower, self.upper)
        return current + np.clip(bounded - current, -self.step, self.step)

    def load_reset_pose(self, path):
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError("复位文件应包含 joints_deg 和 gripper_mm")
        joints = data.get("joints_deg")
        width = data.get("gripper_mm")
        if (
            not isinstance(joints, list)
            or len(joints) != 7
            or any(type(v) not in (int, float) for v in [*joints, width])
        ):
            raise ValueError("复位姿态需要七个关节角（度）和一个夹爪宽度（mm）")
        try:
            goal = np.array([*joints, width], dtype=np.float64)
        except OverflowError as exc:
            raise ValueError("复位姿态超出数值范围") from exc
        goal[:7] = np.deg2rad(goal[:7])
        goal[7] /= 1000
        if not (np.isfinite(goal) & (self.lower <= goal) & (goal <= self.upper)).all():
            raise ValueError("复位姿态含无效数值或超出右臂行程")
        return goal

    def reset_ticks(self, goal, current):
        return max(1.0, float(np.max(np.abs(goal - current) / self.step)))

    def reset_command(self, goal, current):
        # Move all joints along the same joint-space segment, limited by the slowest axis.
        if ((current + self.step < self.lower) | (current - self.step > self.upper)).any():
            raise ValueError("当前姿态超出行程，单条限速指令无法回到有效范围")
        ticks = self.reset_ticks(goal, current)
        if ticks == 1:
            return goal.copy(), True
        target = current + (goal - current) / ticks
        return np.clip(target, self.lower, self.upper), False


# NERO V112 feedback IDs. The passive socket only subscribes to these feedback frames.
CAN_GROUPS = {0x2A5: (0, 2), 0x2A6: (2, 4), 0x2A7: (4, 6), 0x2A9: (6, 7), 0x2A8: (7, 8)}


def passive_feedback(inputs, interface, cancel, max_width):
    frame = struct.Struct("=IB3x8s")
    filters = b"".join(
        struct.pack("=II", can_id, 0xC00007FF) for can_id in (*CAN_GROUPS, *range(0x261, 0x268))
    )
    values = np.zeros(8, dtype=np.float32)
    try:
        with socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW) as bus:
            bus.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, filters)
            bus.settimeout(FEEDBACK_SECONDS)
            bus.bind((interface,))
            fresh = {}
            while not cancel.is_set():
                try:
                    packet = bus.recv(frame.size)
                except TimeoutError:
                    continue
                ident, length, payload = frame.unpack(packet)
                if length == 8 and 0x261 <= ident <= 0x267:
                    with inputs.condition:
                        inputs.motors[ident] = (time.monotonic(), payload[5])
                    continue
                if ident not in CAN_GROUPS or length != 8:
                    continue
                stamp = time.time_ns()
                start, end = CAN_GROUPS[ident]
                if ident == 0x2A8:
                    width = int.from_bytes(payload[:4], "big", signed=True) * 1e-6
                    # In angle mode the ROS driver's stroke-like fallback is confined to range.
                    if payload[7] not in (0, 1) or (
                        payload[7] == 1 and not 0 <= width <= max_width
                    ):
                        fresh.pop(ident, None)
                        continue
                    values[7] = np.clip(width, 0, max_width)
                else:
                    angles = struct.unpack(">ii", payload)[: end - start]
                    values[start:end] = [math.radians(v * 0.001) for v in angles]
                fresh[ident] = stamp
                if len(fresh) == len(CAN_GROUPS):
                    if stamp - min(fresh.values()) <= FEEDBACK_SECONDS * 1e9:
                        inputs.feedback(Feedback(stamp, stamp, values.copy(), "CAN"))
                    fresh.clear()
    except OSError as exc:
        with inputs.condition:
            inputs.events["CAN"] = str(exc)
