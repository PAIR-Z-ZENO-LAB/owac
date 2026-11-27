"""Start a pouring trial at the joint stop, then return control to the visual policy."""

import numpy as np

from owac.runtime.fruit_live import FEEDBACK_SECONDS, HZ


class PouringDeparture:
    def __init__(self, profile, home):
        self.home = np.array(home, dtype=float)
        self.reference = np.asarray(profile["reference_start"], dtype=float)
        self.start_lower = np.minimum(profile["start_lower"], self.home)
        self.start_upper = np.maximum(profile["start_upper"], self.home)
        self.shoulder = np.asarray(profile["shoulder_rad"], dtype=float)
        self.targets = np.asarray(profile["targets_rad_m"], dtype=float)
        self.duration = float(profile["longest_departure_s"])
        if (
            self.home.shape != (8,)
            or self.reference.shape != (8,)
            or not np.all(self.start_lower <= self.start_upper)
            or not np.isfinite(self.reference).all()
            or self.shoulder.ndim != 1
            or len(self.shoulder) < 2
            or self.targets.shape != (len(self.shoulder), 8)
            or not np.isfinite(self.targets).all()
            or not np.all(np.diff(self.shoulder) > 0)
            or not np.all(self.targets[:, 1] > self.shoulder)
            or not 0 < self.duration < np.inf
        ):
            raise ValueError("pouring 起步示范缺少有效的逐点肩关节反馈与目标")
        self.reset()

    def reset(self):
        self.phase = "等待起步"
        self.reached_at = None
        self.hold = None
        self.remaining = 0
        self.suspend()

    def suspend(self):
        self.previous = None
        self.last_at = None

    def command(self, plan, row, count, measured, travel, now, *, rate_limit):
        if self.phase == "等待起步":
            nearest_start = np.clip(measured, self.start_lower, self.start_upper)
            if travel.reset_ticks(nearest_start, measured) == 1:
                self.phase = "示范起步"
                self.offset = measured - self.reference
                self.offset[[1, 7]] = 0
                slew_ticks = travel.reset_ticks(self.targets[-1] + self.offset, measured)
                self.remaining = int(np.ceil(self.duration * HZ + slew_ticks))
            else:
                self.phase = "视觉策略"

        target = plan.actions[row]
        if self.phase == "示范起步":
            if measured[1] >= self.shoulder[-1]:
                self.phase = "等待起步后画面"
                self.reached_at = now
                self.hold = np.clip(measured, travel.lower, travel.upper)
            elif self.remaining == 0:
                raise ValueError("起步时肩关节反馈未到达示范交接位置，请检查右臂控制与现场")
            else:
                # Advance along the demonstrated command curve only with measured motion.
                target = np.array(
                    [np.interp(measured[1], self.shoulder, axis) for axis in self.targets.T]
                )
                target += self.offset
                self.remaining -= 1
        if self.phase == "等待起步后画面":
            if plan.observation.at <= self.reached_at:
                target = self.hold
            else:
                forward = np.clip(plan.actions[:count, 1], travel.lower[1], travel.upper[1])
                if forward.mean() <= measured[1]:
                    raise ValueError("起步后新画面的预测仍指向起点，已保持当前位置")
                self.phase = "视觉策略"

        target = np.clip(target, travel.lower, travel.upper)
        if rate_limit:
            previous = self.previous
            if previous is None or now - self.last_at > FEEDBACK_SECONDS:
                previous = np.clip(measured, travel.lower, travel.upper)
            target[:7] = travel.command(target, previous)[:7]
            self.previous, self.last_at = target.copy(), now
        return target
