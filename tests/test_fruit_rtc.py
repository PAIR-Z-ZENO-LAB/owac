from types import SimpleNamespace

import numpy as np
import pytest

from owac.runtime.fruit_live import Prediction
from owac.runtime.fruit_rtc import PrefixRegression, RegressionRTC, interpolation


def adapter():
    offsets = np.arange(12)
    return PrefixRegression(np.tile(0.8 ** np.abs(offsets[:, None] - offsets), (8, 1, 1)))


def prediction(t, level, key="best"):
    obs = SimpleNamespace(at=t, capture=round(t * 1e9))
    return Prediction(key, obs, np.full((12, 8), level), 0)


def test_conditioning_fractional_commands_changes_the_free_suffix():
    mean = np.zeros((12, 8))
    known = [0.5, 1.8]
    values = np.stack([np.arange(1, 9), np.arange(1, 9) * 3])
    conditioned = adapter().condition(mean, np.arange(12), known, values)
    np.testing.assert_allclose(interpolation(known) @ conditioned, values)
    assert np.all(conditioned[2:] > 0)
    np.testing.assert_allclose(adapter().condition(mean, np.arange(12), [], []), mean)


def test_late_proposal_uses_all_published_limited_commands_at_their_actual_times():
    model = adapter()
    rtc = RegressionRTC({"best": model})
    rtc.start("best", 10.0)
    first, new = prediction(10.002, 1), prediction(10.015, 3)
    now = 10.012
    scaled = np.full(8, 0.75)
    calls = []

    def publish(target):
        issued = np.minimum(target, scaled)
        calls.append((now + 0.001, issued.copy()))
        return now + 0.001, issued

    assert rtc.step(prediction(9.99, 0), 10, publish) is None
    assert rtc.step(first, now, publish)[0] == first.observation.capture
    for proposal in (first, first):
        now = rtc.next_tick + 0.002
        assert rtc.step(proposal, now, publish)[0] == first.observation.capture
    assert rtc.handoffs == 0
    assert rtc.step(new, now + 0.001, publish) is None
    committed = list(rtc.sent)[1:]
    next_time = rtc.next_tick
    position = (next_time - new.observation.at) * 30
    expected = model.condition(
        new.actions,
        [position],
        [(stamp - new.observation.at) * 30 for stamp, _ in calls[1:]],
        [action for _, action in calls[1:]],
        [var for _, _, _, var in committed],
    )[0]
    requested = []
    now = next_time
    capture, _, _ = rtc.step(new, now, lambda x: (now, requested.append(x) or x))
    assert capture == new.observation.capture
    assert rtc.handoffs == rtc.conditioned_handoffs == 1 and rtc.prefix == 2
    for slot, _, sent, _ in committed:
        np.testing.assert_array_equal(rtc.plan.actions[slot - rtc.plan.anchor], sent)
    np.testing.assert_allclose(requested[0], expected)
    assert np.all(expected < new.actions[0])


def test_fast_proposal_has_no_artificially_locked_future_commands():
    rtc = RegressionRTC({"best": adapter()})
    rtc.start("best", 10.0)
    rtc.step(prediction(10.0, 1), 10.0, lambda x: (10.0, x))
    now = rtc.next_tick
    p = prediction(10.020, 3)
    capture, _, applied = rtc.step(p, now, lambda x: (now, x))
    assert capture == p.observation.capture and rtc.prefix == 0
    np.testing.assert_allclose(applied, p.actions[0])


def test_delayed_publication_cannot_repeat_an_already_issued_slot():
    rtc = RegressionRTC({"best": adapter()})
    p = prediction(1, 1)
    rtc.start("best", 1)
    rtc.step(p, 1, lambda x: (1.080, x))
    assert rtc.sent[-1][0] == 2 and rtc.next_tick == 1.1
    assert rtc.step(p, 1.081, lambda _: pytest.fail("duplicate control slot")) is None


def test_control_stall_skips_missed_slots_and_expired_predictions_stop():
    rtc = RegressionRTC({"best": adapter()})
    rtc.start("best", 1)
    old = prediction(1, 1)
    actions = []

    def tick(p, now):
        return rtc.step(p, now, lambda x: (now, actions.append(x) or x))

    assert tick(old, 1)[:2] == (old.observation.capture, 0)
    assert tick(old, 1.101)[:2] == (old.observation.capture, 3)
    assert tick(old, 1.102) is None
    with pytest.raises(RuntimeError, match="过期"):
        tick(old, 1.21)
    assert len(actions) == 2
    rtc.reset()
    assert not rtc.sent and rtc.plan is None
    rtc.start("final", 2)
    with pytest.raises(RuntimeError, match="切换"):
        tick(old, 2)
    assert tick(prediction(1.95, 0, "final"), 2) is None
    with pytest.raises(RuntimeError, match="超时"):
        tick(None, 2.21)
