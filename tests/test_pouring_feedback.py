import json
from types import SimpleNamespace

import numpy as np
import pytest

from owac.runtime.pouring_grip import PouringGrip
from scripts.experiments.summarize_pouring_pause import report


def test_pause_report_keeps_fault_and_excludes_previous_attempt(tmp_path):
    base = dict(policy="owac-feedback", time="18:00", command_file="commands.jsonl")
    pauses = [
        dict(base, sent=1, pause_time_ns=2, reason="手动暂停"),
        dict(base, sent=2, pause_time_ns=4, reason="倾倒时张爪", pouring_fault="张爪"),
        dict(base, sent=2, pause_time_ns=5, reason="退出", pouring_fault="张爪"),
    ]
    rows = [
        dict(time_ns=t, policy=base["policy"], issued_rad_m=[0] * 7 + [0.001 * t])
        for t in (1, 3, 6)
    ]
    for path, entries in (("console-pauses.jsonl", pauses), ("commands.jsonl", rows)):
        (tmp_path / path).write_text("".join(json.dumps(r) + "\n" for r in entries))
    result = "\n".join(report(tmp_path))
    assert "倾倒时张爪" in result and "记录 1 条" in result and "3.0 mm" in result


def guard():
    # Rotate local final z about world y to produce a controllable tilt in the fixture.
    rot = np.repeat(np.eye(3)[None], 7, axis=0)
    rot[-1] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    return PouringGrip(
        dict(
            joint_rotations=rot,
            hand_basis=np.eye(3),
            grasp_gravity_axes=[[-1, 0, 0]],
            empty_m=0.005,
            open_m=0.034,
            tilt_deg=65,
            release_wait_s=0.17,
        )
    )


def joints(width, tilt=0):
    return np.array([0, 0, 0, 0, 0, 0, np.deg2rad(tilt), width])


def test_empty_grasp_stays_latched_until_new_trial_reset():
    g = guard()
    g.check(joints(0.06), joints(0.06), 0)
    with pytest.raises(ValueError, match="空夹"):
        g.check(joints(0), joints(0.001), 0.1)
    g.suspend()
    with pytest.raises(ValueError, match="空夹"):
        g.check(joints(0), joints(0.010), 3)
    g.reset()
    g.check(joints(0), joints(0), 4)


@pytest.mark.parametrize(("goal", "measured"), [(0.05, 0.009), (0, 0.05)])
def test_tilt_stops_on_model_or_measured_opening(goal, measured):
    g = guard()
    g.check(joints(0.06), joints(0.06), 0)
    g.check(joints(0), joints(0.01), 0.1)
    with pytest.raises(ValueError, match="倾倒时"):
        g.check(joints(goal), joints(measured, 90), 0.2)


def test_release_holds_arm_until_measured_opening_and_stops_on_timeout():
    travel = SimpleNamespace(command=lambda target, current: target.copy())
    g = guard()
    g.check(joints(0.06), joints(0.06), 0)
    g.check(joints(0), joints(0.01), 0.1)
    g.check(joints(0), joints(0.01, 90), 0.2)
    release_pose = joints(0.01, 10)
    out = g.command(joints(0.06, 25), release_pose, travel, 0.3)
    np.testing.assert_array_equal(out[:7], release_pose[:7])
    assert out[7] == 0.06
    with pytest.raises(ValueError, match="未跟随"):
        g.command(joints(0.06), joints(0.01), travel, 0.48)


def test_rate_limit_tracks_issued_targets_and_restarts_from_feedback_after_a_gap():
    travel = SimpleNamespace(
        command=lambda target, previous: previous + np.clip(target - previous, -0.02, 0.02)
    )
    g = guard()
    goal, measured = joints(0), joints(0)
    goal[0] = 0.5
    assert g.command(goal, measured, travel, 1)[0] == pytest.approx(0.02)
    assert g.command(goal, measured, travel, 1.04)[0] == pytest.approx(0.04)
    assert g.command(goal, measured, travel, 2)[0] == pytest.approx(0.02)


def test_ros_guard_rejects_before_publication_and_preserves_last_grip_on_pause(monkeypatch):
    pytest.importorskip("rclpy")
    from owac.runtime.fruit_live import Inputs, Prediction
    from owac.runtime.fruit_robot import POLICY, Equipment

    g = guard()
    g.check(joints(0.06), joints(0.06), 0)
    g.check(joints(0), joints(0.009), 0.1)
    inputs = Inputs()
    inputs.key = "candidate"
    fb = SimpleNamespace(values=joints(0.009, 90), source="ROS")
    monkeypatch.setattr(inputs, "at", lambda *args: fb)
    robot = Equipment(
        SimpleNamespace(create_subscription=lambda *args: None),
        inputs,
        None,
        None,
        "unused",
        None,
        20,
        pouring_guards={"candidate": g},
    )
    published = []
    robot.publish = lambda target: published.append(target.copy())
    robot.armed = robot.running = robot.moved = True
    robot.selected = POLICY
    robot.last_command = joints(0.001, 90)
    obs = SimpleNamespace(at=1.0, capture=42, feedback=fb)
    prediction = Prediction("candidate", obs, np.stack([joints(0.06)] * 12), 10)
    with pytest.raises(ValueError, match="倾倒时策略要求张爪"):
        robot.step_chunk("candidate", prediction, obs, 1.05, None, fb.values)
    assert not published and robot.sent == 0
    robot.pause("test")
    np.testing.assert_array_equal(published[0][:7], fb.values[:7])
    assert published[0][7] == 0.001
