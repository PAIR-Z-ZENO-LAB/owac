from types import SimpleNamespace

import numpy as np
import pytest

from owac.runtime.fruit_live import HZ
from owac.runtime.pouring_departure import PouringDeparture

HOME = np.array([0, -1.74, 0, 0, 0, 0, 0, 0.02])
SHOULDER = np.array([-1.74, -1.68, -1.62])
TARGETS = np.tile(HOME, (len(SHOULDER), 1))
TARGETS[:, 1] = SHOULDER + 0.04
PROFILE = dict(
    reference_start=HOME,
    start_lower=HOME - 0.01,
    start_upper=HOME + 0.01,
    shoulder_rad=SHOULDER,
    targets_rad_m=TARGETS,
    longest_departure_s=1.0,
)


class Travel:
    lower = np.array([-2.71, -1.74, -2.75, -1.01, -2.75, -0.73, -1.5707963, 0])
    upper = np.array([2.71, 1.74, 2.75, 2.14, 2.75, 0.95, 1.5707963, 0.10])
    step = np.full(8, 0.03)
    step[-1] = 0.003

    def reset_ticks(self, goal, state):
        return max(1.0, np.max(abs(goal - state) / self.step))

    def command(self, goal, previous):
        return previous + np.clip(
            np.clip(goal, self.lower, self.upper) - previous, -self.step, self.step
        )


def setup():
    travel = Travel()
    home = np.array(PROFILE["reference_start"])
    home = np.clip(home, travel.lower, travel.upper)
    home[1] = travel.lower[1]
    departure = PouringDeparture(PROFILE, home)
    plan = SimpleNamespace(actions=np.tile(home, (12, 1)), observation=SimpleNamespace(at=0.0))
    return travel, home, departure, plan


def test_stationary_feedback_cannot_run_a_blind_departure():
    travel, home, departure, plan = setup()
    issued = []
    with pytest.raises(ValueError, match="肩关节反馈"):
        for i in range(100):
            issued.append(departure.command(plan, 0, 12, home, travel, i / HZ, rate_limit=True))
    assert len(issued) > 1
    assert issued[0][1] > home[1]
    np.testing.assert_allclose(
        issued[-1],
        np.clip(PROFILE["targets_rad_m"][0] + departure.offset, travel.lower, travel.upper),
    )
    assert issued[-1][1] < departure.shoulder[-1]


def test_handoff_uses_new_image_and_does_not_rearm_on_return_home():
    travel, home, departure, plan = setup()
    measured = home.copy()
    for i in range(70):
        target = departure.command(plan, 0, 12, measured, travel, i / HZ, rate_limit=True)
        measured += 0.4 * (target - measured)
        if departure.phase == "等待起步后画面":
            break
    assert departure.phase == "等待起步后画面"
    assert measured[1] >= departure.shoulder[-1]
    plan.observation.at = i / HZ + 0.01
    with pytest.raises(ValueError, match="仍指向起点"):
        departure.command(plan, 0, 12, measured, travel, (i + 1) / HZ, rate_limit=True)
    plan.actions[:] = measured
    plan.actions[:, 1] += 0.05
    departure.command(plan, 0, 12, measured, travel, (i + 2) / HZ, rate_limit=True)
    assert departure.phase == "视觉策略"
    departure.suspend()
    plan.actions[:] = home
    np.testing.assert_array_equal(
        departure.command(plan, 0, 12, home, travel, 10.0, rate_limit=True), home
    )
    departure.reset()
    assert departure.command(plan, 0, 12, home, travel, 11.0, rate_limit=True)[1] > home[1]


def test_starting_away_from_reset_leaves_control_with_the_policy():
    travel, home, departure, plan = setup()
    moved = home.copy()
    moved[0] += 0.4
    plan.actions[:] = moved
    np.testing.assert_array_equal(
        departure.command(plan, 0, 12, moved, travel, 0, rate_limit=True), moved
    )
    assert departure.phase == "视觉策略"
