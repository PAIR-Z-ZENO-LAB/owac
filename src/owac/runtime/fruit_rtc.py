"""Real-time chunk execution for the deterministic OWAC fruit regressors."""

import hashlib
import json
from collections import deque
from dataclasses import dataclass

import numpy as np

from owac.runtime.fruit_live import FEEDBACK_SECONDS, HZ

HORIZON = 12
ACTION_DIM = 8
NS = 10**9


def interpolation(times):
    """Linear observations of a 30 Hz chunk at fractional sample times."""
    times = np.asarray(times, dtype=float)
    if np.any((times < 0) | (times > HORIZON - 1)):
        raise ValueError("RTC query outside the action horizon")
    return np.maximum(1 - np.abs(times[:, None] - np.arange(HORIZON)), 0)


class PrefixRegression:
    """Conditional mean of a regression chunk given commands already sent on its timeline.

    The checkpoint supplies the mean. The fitted residual covariance spans the 12 times
    of each physical actuator. Conditioning on a prefix changes the free suffix, even
    when none of the old plan's unexecuted targets is available.
    """

    def __init__(self, covariance):
        self.cov = np.asarray(covariance, dtype=np.float64)
        if self.cov.shape != (ACTION_DIM, HORIZON, HORIZON) or not np.isfinite(self.cov).all():
            raise ValueError("Invalid RTC calibration for 12 time steps and 8 actuators")

    @classmethod
    def load(cls, path, checkpoint, step):
        with path.open() as src:
            model = json.load(src)
        with checkpoint.open("rb") as src:
            digest = hashlib.file_digest(src, "sha256").hexdigest()
        if (model["version"], model["step"], model["frequency_hz"], model["checkpoint_sha256"]) != (
            1,
            step,
            HZ,
            digest,
        ):
            raise ValueError(f"RTC 参数与所选权重不匹配 {path.name}")
        return cls(model["covariance"])

    def condition(self, mean, requested, committed, sent, variance=None):
        query = interpolation(requested)
        result = query @ mean
        if len(committed):
            observed = interpolation(committed)
            cross = self.cov @ observed.T
            observed_cov = observed @ cross
            if variance is not None:
                ids = np.arange(len(committed))
                observed_cov[:, ids, ids] += np.asarray(variance).T
            gain = (query @ cross) @ np.linalg.pinv(observed_cov, hermitian=True)
            difference = np.asarray(sent) - observed @ mean
            result += np.einsum("dhp,pd->hd", gain, difference)
        result[:, -1] = np.maximum(0, result[:, -1])
        return result

    def command_variance(self, mean, at, issued):
        # An old policy command is an uncertain estimate of a demonstration action.
        # Its squared error includes both model variance and the actual limiting/correction.
        h = interpolation([at])[0]
        return (h @ self.cov @ h) + (h @ mean - issued) ** 2


@dataclass
class Chunk:
    prediction: object
    anchor: int
    actions: np.ndarray


class RegressionRTC:
    """The control thread owns the queue. The GPU worker can finish at any time."""

    def __init__(self, adapters):
        self.adapters = adapters
        self.reset()

    def reset(self):
        self.epoch = None
        self.epoch_ns = 0
        self.key = None
        self.plan = None
        self.sent = deque(maxlen=HORIZON)
        self.handoffs = self.conditioned_handoffs = self.prefix = 0
        self.next_tick = 0.0

    def start(self, key, now):
        self.reset()
        self.key, self.epoch, self.next_tick = key, now, now
        self.epoch_ns = round(now * NS)

    def _deadline_ns(self, slot):
        return self.epoch_ns + (slot * NS + HZ - 1) // HZ

    def _install(self, proposal, slot):
        obs_ns = round(proposal.observation.at * NS)
        anchor = -((self.epoch_ns - obs_ns) * HZ // NS)
        offsets = np.array([(self._deadline_ns(anchor + i) - obs_ns) * HZ for i in range(HORIZON)])
        grid = offsets[offsets <= (HORIZON - 1) * NS] / NS
        if not 0 <= slot - anchor < len(grid):
            return False
        history = [
            (s, (t - obs_ns) * HZ / NS, action, variance)
            for s, t, action, variance in self.sent
            if 0 <= (t - obs_ns) * HZ <= (HORIZON - 1) * NS
        ]
        times = [t for _, t, _, _ in history]
        values = [action for _, _, action, _ in history]
        variances = [var for _, _, _, var in history]
        actions = self.adapters[self.key].condition(
            proposal.actions, grid, times, values, variances
        )
        # Retain exactly what was published in committed control slots, including jitter and limits.
        for s, _, action, _ in history:
            if 0 <= s - anchor < len(grid):
                actions[s - anchor] = action
        self.prefix = len(history)
        self.conditioned_handoffs += int(self.plan is not None and bool(history))
        self.handoffs += int(self.plan is not None)
        self.plan = Chunk(proposal, anchor, actions)
        return True

    def step(self, proposal, now, send):
        """Publish one due slot and retain its actual post-limit command."""
        if self.epoch is None:
            raise RuntimeError("RTC 尚未启动")
        if now < self.next_tick:
            return None
        now_ns = round(now * NS)
        slot = (now_ns - self.epoch_ns) * HZ // NS
        if self.sent and slot <= self.sent[-1][0]:
            return None
        if proposal is not None and proposal.checkpoint != self.key:
            raise RuntimeError("RTC 权重已切换，请重新开始")
        usable = (
            proposal is not None
            and self.epoch <= proposal.observation.at <= now
            and now - proposal.observation.at <= FEEDBACK_SECONDS
        )
        if self.plan is None:
            # Phase-lock to the first dispatch, so a camera arriving just after the
            # start key never loses its first executable control slot.
            self.epoch_ns = now_ns
            slot = 0
            if not usable or not self._install(proposal, slot):
                if now - self.epoch > FEEDBACK_SECONDS:
                    raise RuntimeError("RTC 等待本次启动后的新预测超时")
                return None
        elif not usable:
            raise RuntimeError("RTC 图像或预测已过期")
        if self.sent and (slot - self.sent[-1][0]) / HZ > FEEDBACK_SECONDS:
            raise RuntimeError("RTC 控制时钟中断，请重新开始")

        plan = self.plan
        row = slot - plan.anchor
        newer = proposal.observation.capture > plan.prediction.observation.capture
        time_left = now - plan.prediction.observation.at <= FEEDBACK_SECONDS
        expired = not time_left or not 0 <= row < len(plan.actions)
        if newer and self._install(proposal, slot):
            plan = self.plan
            row = slot - plan.anchor
            newer = False
        elif expired:
            raise RuntimeError("RTC 队列已过期或耗尽")

        # Record what actually went out. Only future handoffs can consume this command.
        sent_at, applied = send(plan.actions[row])
        command = np.asarray(applied, dtype=np.float64).copy()
        offset = (self._deadline_ns(slot) - round(plan.prediction.observation.at * NS)) * HZ / NS
        variance = self.adapters[self.key].command_variance(
            plan.prediction.actions, offset, command
        )
        sent_ns = round(sent_at * NS)
        issued_slot = (sent_ns - self.epoch_ns) * HZ // NS
        self.sent.append((issued_slot, sent_ns, command, variance))
        self.next_tick = self._deadline_ns(issued_slot + 1) / NS
        if newer:
            # Its observation arrived after this slot's nominal time. Stage it for the next slot.
            self._install(proposal, issued_slot + 1)
        return plan.prediction.observation.capture, row, command
