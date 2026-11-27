"""First-placement gripper recovery and joint-only smoothing for the three-bowl policy."""

import json
from collections import deque
from pathlib import Path

import numpy as np

from owac.runtime.fruit_live import HZ


def hand_center(joints, geometry):
    """NERO's two-finger midpoint, using the calibrated URDF chain in the profile."""
    q = np.asarray(joints)[..., :7]
    c, s = np.cos(q), np.sin(q)
    z, one = np.zeros_like(c), np.ones_like(c)
    turn = np.stack((c, -s, z, s, c, z, z, z, one), axis=-1).reshape(*q.shape, 3, 3)
    offsets, rotations, tip = (np.asarray(geometry[k]) for k in ("xyz", "rot", "tip"))
    rotation = rotations[0] @ turn[..., 0, :, :]
    position = offsets[0]
    for i in range(1, 7):
        position = position + rotation @ offsets[i]
        rotation = (rotation @ rotations[i]) @ turn[..., i, :, :]
    return position + rotation @ tip


def motion_features(xyz, anchor, peak, velocity):
    delta = xyz - anchor
    return np.stack(
        (
            *np.moveaxis(delta, -1, 0),
            np.linalg.norm(delta[..., :2], axis=-1),
            np.broadcast_to(peak - anchor[..., 2], delta.shape[:-1]),
            peak - xyz[..., 2],
            np.linalg.norm(velocity[..., :2], axis=-1),
            velocity[..., 2],
            np.linalg.norm(velocity, axis=-1),
        ),
        axis=-1,
    )


class JointPlanSmoother:
    """Track each short OWAC plan with a penalty on changes in joint velocity."""

    def __init__(self, penalty, limits):
        self.penalty = float(penalty)
        self.lower, self.upper = np.asarray(limits)
        self.reset()

    def reset(self):
        self.last = None
        self.last_time = None
        self.velocity = np.zeros(7)
        self.plan = None

    def step(self, targets, measured, now, row):
        if row == 0 or self.plan is None:
            previous = np.asarray(measured)[:7] if self.last is None else self.last
            dt = (
                1
                if self.last_time is None or now <= self.last_time
                else (now - self.last_time) * HZ
            )
            n = len(targets)
            # Differences use the actual interval across plans and the 30 Hz plan clock thereafter.
            d = np.eye(n) - 2 * np.eye(n, k=-1) + np.eye(n, k=-2)
            b = np.zeros((n, 7))
            d[0, 0] = 1 / dt
            b[0] = previous / dt + self.velocity
            if n > 1:
                d[1, 0] = -(1 + 1 / dt)
                b[1] = -previous / dt
            self.plan = np.linalg.solve(
                np.eye(n) + self.penalty * (d.T @ d),
                np.asarray(targets)[:, :7] + self.penalty * (d.T @ b),
            )
        result = np.clip(self.plan[row], self.lower, self.upper)
        if self.last is not None and now > self.last_time:
            self.velocity = (result - self.last) / ((now - self.last_time) * HZ)
        self.last, self.last_time = result.copy(), now
        return result


class BowlPostprocessor:
    """Use measured grasp history and the current RAW scene to recover a missing first release."""

    def __init__(self, profile):
        self.profile = profile
        self.models = {
            name: (np.asarray(row["motion"]), np.asarray(row["scene"]), row["bias"])
            for name, row in profile["events"].items()
        }
        self.arm = JointPlanSmoother(profile["joint_penalty"], profile["joint_limits_rad"])
        self.reset()

    @classmethod
    def load(cls, path):
        profile = json.loads(Path(path).read_text())
        if profile.get("kind") == "bowl_flow_v1":
            from owac.runtime.bowl_flow import BowlFlow

            return BowlFlow.load(path)
        return cls(profile)

    def reset(self):
        self.arm.reset()
        self.phase = "before_grasp"
        self.saw_open = False
        self.anchor = None
        self.peak = None
        self.score = None
        self.samples = deque()

    def suspend(self):
        self.arm.reset()
        self.samples.clear()

    def enter(self, phase, xyz):
        self.phase = phase
        self.anchor = xyz.copy()
        self.peak = xyz[2]

    def jaw(self, raw, measured, scene, now):
        threshold = self.profile["width_split_m"]
        self.score = None
        if self.phase == "native":
            return raw
        xyz = hand_center(measured[:7], self.profile["geometry"])
        self.samples.append((now, xyz))
        cutoff = now - self.profile["motion_window_s"]
        while len(self.samples) > 1 and self.samples[1][0] <= cutoff:
            self.samples.popleft()
        at, old = self.samples[0]
        velocity = (xyz - old) / (now - at) if now > at else np.zeros(3)
        if self.phase == "before_grasp":
            if measured[7] > threshold:
                self.saw_open = True
            elif self.saw_open:
                self.enter("holding", xyz)
            return raw
        if self.phase == "opening":
            if measured[7] > threshold:
                self.enter("after_release", xyz)
            return self.profile["open_width_m"]
        if self.phase == "holding" and measured[7] > threshold:
            # The native model already opened successfully before any intervention.
            self.phase = "native"
            return raw
        self.peak = max(self.peak, xyz[2])
        if scene is not None:
            event = "release" if self.phase == "holding" else "next_grasp"
            motion, vision, bias = self.models[event]
            x = motion_features(xyz, self.anchor, self.peak, velocity)
            self.score = float(x @ motion + np.asarray(scene) @ vision + bias)
            if self.score >= 0:
                if self.phase == "holding":
                    self.phase = "opening"
                    return max(raw, self.profile["open_width_m"])
                self.phase = "native"
                return raw
        return max(raw, self.profile["open_width_m"]) if self.phase == "after_release" else raw

    def command(self, plan, row, measured, scene, now):
        target = np.asarray(plan[row], dtype=float).copy()
        target[:7] = self.arm.step(plan, measured, now, row)
        target[7] = self.jaw(target[7], measured, scene, now)
        return target


def process_prediction(processor, prediction, count, row, measured, now):
    if hasattr(processor, "observe_frame"):
        processor.observe_frame(prediction.observation)
    return processor.command(prediction.actions[:count], row, measured, prediction.scene, now)
