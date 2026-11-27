"""Stop a pouring attempt on measured grip loss or an opening request during tilt."""

import numpy as np

from owac.runtime.fruit_live import FEEDBACK_SECONDS


class PouringGrip:
    def __init__(self, profile):
        self.profile = profile
        self.rotations = np.asarray(profile["joint_rotations"])
        self.hand_basis = np.asarray(profile["hand_basis"])
        self.upright = np.asarray(profile["grasp_gravity_axes"])
        self.reset()

    def reset(self):
        self.phase = "等待抓杯前张爪"
        self.fault = None
        self.angle = None
        self.release_since = None
        self.release_pose = None
        self.grip_seen = False
        self.last_target = None
        self.last_at = None

    def suspend(self):
        self.release_since = None
        self.last_target = None
        self.last_at = None

    @property
    def carrying(self):
        return self.phase in ("闭爪抓杯", "倾倒与回正", "等待放杯张开反馈")

    def pose_angle(self, joints):
        # World vertical expressed in the wrist frame. Compare with upright grasps
        # from the training demos, allowing variation in approach and cup handle pose.
        vertical = np.array([0.0, 0.0, 1.0])
        for origin, joint in zip(self.rotations, joints[:7], strict=True):
            x, y, z = vertical @ origin
            c, s = np.cos(joint), np.sin(joint)
            vertical = np.array([x * c + y * s, -x * s + y * c, z])
        cosines = self.upright @ (vertical @ self.hand_basis)
        return float(np.rad2deg(np.arccos(np.clip(cosines.max(), -1, 1))))

    def check(self, goal, measured, now):
        if self.fault:
            raise ValueError(self.fault)
        self.angle = self.pose_angle(measured)
        open_at = self.profile["open_m"]
        target, width = float(goal[7]), float(measured[7])
        tilted = self.angle >= self.profile["tilt_deg"]
        if self.phase == "等待抓杯前张爪":
            if min(target, width) >= open_at:
                self.phase = "已张爪接近杯"
        elif self.phase == "已张爪接近杯":
            if target < open_at:
                self.phase = "闭爪抓杯"
        if self.carrying:
            if self.profile["empty_m"] <= width < open_at and target < open_at:
                self.grip_seen = True
            if tilted and target >= open_at:
                self.fault = "倾倒时策略要求张爪"
            elif tilted and width >= open_at:
                self.fault = "倾倒时反馈显示夹爪张开"
            elif width < self.profile["empty_m"] and target < open_at:
                self.fault = "闭爪反馈进入空夹范围"
            elif tilted:
                self.phase = "倾倒与回正"
            elif self.phase == "闭爪抓杯" and self.grip_seen and target >= open_at:
                self.fault = "尚未观察到示范中的倾倒幅度，策略已要求放杯"
            elif self.phase == "倾倒与回正" and target >= open_at:
                self.phase = "等待放杯张开反馈"
                self.release_pose = np.array(measured[:7], copy=True)
            if self.phase == "等待放杯张开反馈":
                if width >= open_at:
                    self.phase = "已观察到放杯张爪"
                elif self.release_since is None:
                    self.release_since = now
                elif now - self.release_since > self.profile["release_wait_s"]:
                    self.fault = "放杯时已请求张爪，实测开度未跟随"
        if self.fault:
            self.fault += (
                f"（腕部倾角 {self.angle:.1f}°，实测 {width * 1000:.1f} mm，"
                f"目标 {target * 1000:.1f} mm）。已暂停，请检查杯与现场"
            )
            raise ValueError(self.fault)

    def command(self, goal, measured, travel, now):
        if self.last_at is not None and now - self.last_at > FEEDBACK_SECONDS:
            self.suspend()
        self.check(goal, measured, now)
        target = goal.copy()
        if self.phase == "等待放杯张开反馈":
            target[:7] = self.release_pose
        # Use the console's deployment velocity envelope for the arm. The gripper
        # keeps the learned target and the native measured 0..100 mm width limit.
        previous = measured if self.last_target is None else self.last_target
        target[:7] = travel.command(target, previous)[:7]
        self.last_target = target.copy()
        self.last_at = now
        return target

    def diagnostics(self):
        return dict(pouring_phase=self.phase, wrist_tilt_deg=self.angle, pouring_fault=self.fault)
