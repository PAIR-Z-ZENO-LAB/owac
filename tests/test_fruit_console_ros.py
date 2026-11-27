"""Exercise the installed NERO mux in a private ROS namespace using a simulated follower."""

import json
import os
import time
from types import SimpleNamespace

import numpy as np
import pytest

from owac.runtime.fruit_live import Feedback, Inputs, Prediction, Travel
from owac.runtime.fruit_rtc import PrefixRegression, RegressionRTC


@pytest.mark.parametrize("rtc_enabled", [False, True], ids=["direct", "rtc"])
def test_prediction_arriving_during_a_control_tick(monkeypatch, rtc_enabled):
    pytest.importorskip("rclpy")
    from owac.runtime import fruit_robot as hw

    tick_at = 100.0
    wall = time.time_ns()
    feedback = Feedback(wall, wall, np.zeros(8), "ROS")
    inputs = Inputs()
    inputs.select("best")
    inputs.feedback(feedback)
    rtc = RegressionRTC({"best": PrefixRegression(np.tile(np.eye(12), (8, 1, 1)))})
    node = SimpleNamespace(create_subscription=lambda *args: None)
    travel = SimpleNamespace(command=lambda target, current: target.copy())
    robot = hw.Equipment(node, inputs, None, None, "test-only", travel, 20, rtc=rtc)
    robot.armed = robot.running = True
    robot.selected = hw.POLICY
    robot.rtc_enabled = rtc_enabled
    if rtc_enabled:
        rtc.start("best", tick_at - 0.05)
    published = []
    monkeypatch.setattr(robot, "publish", lambda target: published.append(target.copy()))

    # The UI entered the tick first. The camera frame and its GPU result arrived
    # while the control thread was checking the ROS graph.
    clock = SimpleNamespace(monotonic=lambda: tick_at + 0.030, time_ns=lambda: wall)
    monkeypatch.setattr(hw, "time", clock)

    def check_health(now):
        obs = SimpleNamespace(at=tick_at + 0.010, capture=1, feedback=feedback)
        inputs.prediction = Prediction("best", obs, np.ones((12, 8)), 10.0)
        return ""

    monkeypatch.setattr(robot, "health", check_health)
    robot.tick(tick_at)
    assert robot.running, robot.info
    assert robot.sent == 1 and len(published) == 1
    np.testing.assert_array_equal(published[0], np.ones(8))

    # Conversely a frame that aged out while this tick waited must only issue a hold.
    clock.monotonic = lambda: tick_at + 0.250
    robot.tick(tick_at + 0.070)
    assert not robot.running and robot.sent == 1
    assert "240.0 ms" in robot.info and "上限 200 ms" in robot.info
    assert len(published) == 2 and not rtc.sent
    np.testing.assert_array_equal(published[1], feedback.values)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("exercise_reset", "direct"),
    [(False, False), (True, False), (False, True)],
    ids=["policy", "reset", "direct"],
)
def test_mux_step_run_pause_and_stale_input(monkeypatch, tmp_path, exercise_reset, direct):
    rclpy = pytest.importorskip("rclpy")
    from agx_arm_msgs.msg import AgxArmStatus, GripperStatus
    from piper_nero_teleop.joint_cmd_mux_node import JointCmdMuxNode
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import JointState

    from owac.runtime import fruit_robot as hw
    from owac.runtime import nero_controllers as teleop

    # Namespaced ROS messages only. The test never starts a native motor driver or opens a CAN bus.
    ns = f"/fruit_tui_test_{os.getpid()}/slave_right"
    policy = ns + "/vla_joint_cmd"
    for key, val in {
        "NS": ns,
        "POLICY": policy,
        "MUX": ns + "/joint_cmd_mux",
        "STATE": ns + "/joint_states_single",
        "FOLLOWER": ("nero_ctrl_node", ns),
        "ROUTER": ("joint_cmd_mux_node", ns),
    }.items():
        monkeypatch.setattr(hw, key, val)
    monkeypatch.setenv("ROS_DOMAIN_ID", "83")
    monkeypatch.setenv("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")
    monkeypatch.setattr(
        "owac.runtime.nero_controllers.local_controllers", lambda: {(ns, "test-only")}
    )
    args = ["--ros-args", "-r", f"joint_cmd_mux_node:__ns:={ns}"]
    for key, val in {
        "teleop_joint_topic": ns + "/mock_master",
        "teleop_gripper_topic": ns + "/mock_grip",
        "policy_topic": policy,
        "global_selected_topic": ns + "/selected",
    }.items():
        args += ["-p", f"{key}:={val}"]
    rclpy.init(args=args)
    executor = SingleThreadedExecutor()
    mux = JointCmdMuxNode()
    follower = rclpy.create_node("nero_ctrl_node", namespace=ns)
    console = rclpy.create_node("local_console", namespace=ns)
    for node in (mux, follower, console):
        executor.add_node(node)
    pos = np.array([0.1] * 7 + [0.04], dtype=np.float32).astype(float).tolist()
    state = follower.create_publisher(JointState, hw.STATE, 1)
    arm = follower.create_publisher(AgxArmStatus, ns + "/arm_status", 1)
    grip = follower.create_publisher(GripperStatus, ns + "/gripper_status", 1)
    output = []
    tracking = False

    def receive(msg):
        output.append(list(msg.position))
        if tracking:
            measured = np.asarray(msg.position, dtype=np.float32)
            measured[0] += np.deg2rad(1.0)  # Tracking residual at the final NERO waypoint.
            pos[:] = measured.astype(float).tolist()

    follower.create_subscription(JointState, hw.MUX + "/output", receive, 1)
    inputs = Inputs()
    travel = Travel(teleop.ROOT / "agx_arm_urdf/nero/urdf", 0.2)
    reset_pose = np.array([0.21, 0.18, -0.01, 0.31, 0.22, 0.12, 0.15, 0.025])
    rtc = RegressionRTC({"best": PrefixRegression(np.tile(np.eye(12), (8, 1, 1)))})
    robot = hw.Equipment(
        console,
        inputs,
        None,
        tmp_path,
        "test-only",
        travel,
        20,
        reset_pose=reset_pose,
        rtc=rtc,
        direct=direct,
        trace_mux=direct,
    )
    monkeypatch.setattr(
        robot, "start_child", lambda *args: pytest.fail("hardware launch attempted")
    )
    if not exercise_reset:
        inputs.select("best")
    desired = np.array([0.11] * 7 + [0.042])
    current = Feedback(time.time_ns(), time.time_ns(), np.array(pos), "ROS")
    stale_camera = False

    def telemetry():
        msg = JointState(name=list(hw.JOINTS), position=pos)
        msg.header.stamp = follower.get_clock().now().to_msg()
        state.publish(msg)
        arm.publish(AgxArmStatus())
        grip.publish(GripperStatus(width=pos[7], mode="width", driver_enable_status=True))
        now = time.monotonic()
        inputs.motors = {k: (now, 0x40) for k in range(7)}
        if inputs.key is None:
            return
        obs = SimpleNamespace(
            capture=time.monotonic_ns(), at=now - (1 if stale_camera else 0.045), feedback=current
        )
        inputs.observation = obs
        inputs.prediction = Prediction(inputs.key, obs, np.tile(desired, (12, 1)), 10)

    follower.create_timer(0.01, telemetry)

    def spin_until(predicate, timeout=5):
        end = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() > end:
                pytest.fail(f"Timed out: {robot.info}, {robot.selected}, {len(output)}")
            executor.spin_once(timeout_sec=0.01)
            robot.tick(time.monotonic())

    try:
        spin_until(lambda: inputs.at("ROS", time.time_ns()) is not None and bool(robot.selected))
        robot.step(time.monotonic())
        assert not output and robot.sent == 0
        assert console.count_publishers(policy) == 0
        robot.connect()
        spin_until(lambda: robot.armed)
        assert output and all(np.array_equal(item, pos) for item in output)
        if direct:
            # A target from the live banana scene. Only this private mock follower receives it.
            desired = np.r_[np.deg2rad([0.65, -104.9, -9.85, 123.94, -7.19, 15.77, 93.43]), 0.00442]
            assert not np.array_equal(travel.command(desired, np.array(pos)), desired)
            spin_until(
                lambda: (
                    np.array_equal(inputs.prediction.actions[0], desired)
                    and time.monotonic() >= robot.next_tick
                )
            )
            robot.step(time.monotonic(), continuous=False)
            spin_until(lambda: np.array_equal(output[-1], desired))
            assert robot.sent == 1 and not robot.running and not robot.rtc_enabled
            robot.step(time.monotonic(), continuous=True)
            spin_until(lambda: robot.sent >= 5)
            assert all(np.array_equal(v, desired) for v in output[-3:])
            spin_until(
                lambda: robot.last_mux is not None and np.allclose(robot.last_mux[1], desired)
            )
            assert robot.last_mux[0] > 0
            robot.pause("operator pause")
            spin_until(lambda: np.array_equal(output[-1], pos))
            assert robot.last_policy_target[0] == "best"
            np.testing.assert_array_equal(robot.last_policy_target[1], desired)
            event = json.loads((tmp_path / "console-pauses.jsonl").read_text().splitlines()[-1])
            assert event["direct"] and not event["rtc_enabled"]
            np.testing.assert_array_equal(event["last_policy_target_rad_m"], desired)
            return
        if exercise_reset:
            assert inputs.key is None and inputs.prediction is None
            robot.start_reset(time.monotonic())
            robot.step(time.monotonic(), continuous=True)
            spin_until(lambda: any(not np.array_equal(item, pos) for item in output))
            assert robot.reset_deadline is not None and not robot.running and robot.sent == 0
            robot.pause("operator pause")
            spin_until(lambda: np.array_equal(output[-1], pos))
            assert robot.reset_deadline is None

            # A responsive driver with a stationary arm must not be reported as having arrived.
            robot.start_reset(time.monotonic())
            assert robot.reset_deadline is not None
            robot.reset_deadline = time.monotonic()
            spin_until(lambda: robot.reset_deadline is None)
            assert "超时" in robot.info

            tracking = True
            start_at = len(output)
            robot.start_reset(time.monotonic())
            spin_until(lambda: robot.reset_deadline is None)
            assert "复位完成" in robot.info
            np.testing.assert_allclose(pos, reset_pose + np.r_[np.deg2rad(1), np.zeros(7)])
            np.testing.assert_allclose(output[-1], reset_pose)
            assert len(output[start_at:]) > 2 and not robot.running and robot.sent == 0
            inputs.select("best")
            inputs.prediction = Prediction(
                "best",
                SimpleNamespace(at=robot.policy_after - 0.005, capture=1, feedback=current),
                np.tile(desired, (12, 1)),
                1,
            )
            robot.step(time.monotonic(), continuous=True)
            assert not robot.running and robot.sent == 0 and "新画面" in robot.info
            tracking = False
            desired = np.array(pos)
            desired[:7] += 0.005
            desired[7] += 0.001
            spin_until(lambda: inputs.prediction.observation.at > robot.policy_after)
        spin_until(lambda: time.monotonic() >= robot.next_tick)
        robot.step(time.monotonic())
        spin_until(lambda: any(np.array_equal(item, desired) for item in output))
        assert robot.sent == 1 and not robot.running
        robot.step(time.monotonic(), continuous=True)
        spin_until(lambda: robot.sent >= 6)
        assert rtc.handoffs and rtc.prefix and robot.rtc_enabled
        robot.toggle_rtc()
        stopped = robot.sent
        spin_until(lambda: np.array_equal(output[-1], pos))
        assert not robot.running and robot.sent == stopped and rtc.plan is None
        assert not robot.rtc_enabled
        robot.step(time.monotonic(), continuous=True)
        spin_until(lambda: robot.sent > stopped)
        robot.toggle_rtc()
        robot.step(time.monotonic(), continuous=True)
        spin_until(lambda: rtc.handoffs)
        assert rtc.sent and robot.rtc_enabled
        if exercise_reset:
            robot.start_reset(time.monotonic())
            assert rtc.plan is None and not rtc.sent
            robot.pause("operator pause")
            robot.step(time.monotonic(), continuous=True)
            spin_until(lambda: inputs.prediction.observation.at > robot.policy_after)
            robot.step(time.monotonic(), continuous=True)
            spin_until(lambda: rtc.plan is not None)
        stale_camera = True
        spin_until(lambda: not robot.running)
        assert rtc.plan is None and not rtc.sent
        assert robot.selected == policy
        event = json.loads((tmp_path / "console-pauses.jsonl").read_text().splitlines()[-1])
        assert event["reason"] == robot.info
        assert "上限 200 ms" in robot.info
        assert event["rtc_enabled"] and event["running"]
        assert event["prediction_frame_ms"] >= 1000
    finally:
        robot.close()
        executor.shutdown()
        for n in (console, follower, mux):
            n.destroy_node()
        rclpy.shutdown()


@pytest.mark.parametrize("horizon", [None, 4], ids=["complete", "four-commands"])
def test_policy_executes_selected_rows_and_replans_from_post_action_feedback(monkeypatch, horizon):
    pytest.importorskip("rclpy")
    from owac.runtime import fruit_robot as hw

    inputs = Inputs()
    inputs.select("banana")
    robot = hw.Equipment(
        SimpleNamespace(create_subscription=lambda *args: None),
        inputs,
        None,
        None,
        "test-only",
        None,
        20,
        run_chunks=True,
        execution_horizons={"banana": horizon} if horizon else None,
    )
    robot.armed = True
    robot.selected = hw.POLICY
    robot.health = lambda now: ""
    published = []
    robot.publish = lambda q: published.append(q.copy())
    now = 120.0
    clock = SimpleNamespace(monotonic=lambda: now, time_ns=lambda: round(now * 1e9))
    monkeypatch.setattr(hw, "time", clock)

    def update(capture, age=0.01):
        wall = clock.time_ns()
        fb = Feedback(wall, wall, np.zeros(8), "ROS")
        inputs.feedback(fb)
        obs = SimpleNamespace(at=now - age, capture=capture, feedback=fb)
        inputs.observation = obs
        actions = np.arange(96).reshape(12, 8) / 1000 + capture
        inputs.prediction = Prediction("banana", obs, actions, 10)
        return actions

    update(0)
    robot.step(now, continuous=True)
    assert robot.running and not published
    count = horizon or 12
    for capture in range(1, count + 1):
        now += 1 / 30 + 0.001
        proposal = update(capture)
        if capture == 1:
            first = proposal
        robot.step(now)
    np.testing.assert_array_equal(published, first[:count])
    # A late result from before the last action must not start the next chunk.
    now += 1 / 30 + 0.001
    update(13, 0.08)
    robot.step(now)
    assert robot.sent == count and robot.running
    now += 1 / 30 + 0.001
    second = update(14)
    robot.step(now)
    np.testing.assert_array_equal(published[-1], second[0])
    assert robot.sent == count + 1
    robot.pause("operator")
    assert robot.chunk is None and not robot.running


def test_non_rtc_runner_uses_a_valid_later_row_when_latest_camera_is_fresh(monkeypatch):
    pytest.importorskip("rclpy")
    from owac.runtime import fruit_robot as hw

    now = 50.0
    wall = round(now * 1e9)
    inputs = Inputs()
    inputs.select("best")
    fb = Feedback(wall, wall, np.zeros(8), "ROS")
    inputs.feedback(fb)
    # The failure report measured 19.7 ms for the newest frame and 220.4 ms
    # for the prediction. Its seventh interval is still inside the trained plan.
    inputs.observation = SimpleNamespace(at=now - 0.0197, capture=2, feedback=fb)
    plan = np.arange(96).reshape(12, 8) / 1000.0
    inputs.prediction = Prediction(
        "best", SimpleNamespace(at=now - 0.2204, capture=1, feedback=fb), plan, 100.9
    )
    robot = hw.Equipment(
        SimpleNamespace(create_subscription=lambda *args: None),
        inputs,
        None,
        None,
        "test-only",
        SimpleNamespace(command=lambda q, measured: q),
        20,
    )
    robot.armed = True
    robot.selected = hw.POLICY
    robot.health = lambda _: ""
    published = []
    robot.publish = lambda q: published.append(q.copy())
    monkeypatch.setattr(hw, "time", SimpleNamespace(monotonic=lambda: now, time_ns=lambda: wall))
    robot.step(now, continuous=True)
    assert robot.sent == 1 and robot.running
    np.testing.assert_array_equal(published[0], plan[7])


@pytest.mark.parametrize("chunks", [True, False], ids=["chunk", "direct"])
def test_bowl_policy_sends_a_physical_width_and_logs_the_original_prediction(monkeypatch, chunks):
    from io import StringIO

    pytest.importorskip("rclpy")
    from owac.runtime import fruit_robot as hw

    inputs = Inputs()
    inputs.select("owac")
    stream = StringIO()
    bounds = SimpleNamespace(lower=np.zeros(8), upper=np.r_[np.ones(7), 0.10])
    robot = hw.Equipment(
        SimpleNamespace(create_subscription=lambda *args: None),
        inputs,
        None,
        None,
        "test-only",
        bounds,
        20,
        direct=not chunks,
        run_chunks=chunks,
        trace_mux=True,
        physical_gripper_width=True,
        command_log=stream,
    )
    robot.armed = True
    robot.selected = hw.POLICY
    robot.health = lambda now: ""
    issued = []
    robot.publish = lambda target: issued.append(target.copy())
    clock = SimpleNamespace(monotonic=lambda: now, time_ns=lambda: round(now * 1e9))
    monkeypatch.setattr(hw, "time", clock)
    for i, (width, expected) in enumerate([(-0.009, 0.0), (0.053, 0.053), (0.103, 0.10)]):
        now = 100.0 + i
        wall = clock.time_ns()
        state = Feedback(wall, wall, np.zeros(8), "ROS")
        inputs.feedback(state)
        obs = SimpleNamespace(at=now, capture=i + 1, feedback=state)
        inputs.observation = obs
        raw = np.r_[np.linspace(-2, 2, 7), width]
        inputs.prediction = Prediction("owac", obs, np.tile(raw, (12, 1)), 10)
        robot.step(now, continuous=False)
        np.testing.assert_array_equal(issued[-1], np.r_[raw[:7], expected])
        np.testing.assert_array_equal(robot.last_model_target, raw)
        np.testing.assert_array_equal(robot.last_policy_target[1], issued[-1])
        record = json.loads(stream.getvalue().splitlines()[-1])
        assert record["model_rad_m"][7] == width and record["issued_rad_m"][7] == expected
        assert record["feedback_ns"] == wall and record["row"] == 0
    assert robot.sent == len(issued) == 3


def test_bowl_chunk_publishes_processed_motion_and_keeps_native_in_log(monkeypatch):
    from io import StringIO
    from pathlib import Path

    pytest.importorskip("rclpy")
    from owac.runtime import fruit_robot as hw
    from owac.runtime.bowl_postprocess import BowlPostprocessor

    cfg = json.loads((Path(__file__).resolve().parents[1] / "tests/fixtures/bowl.json").read_text())
    proc = BowlPostprocessor(cfg)
    proc.models["release"] = (np.zeros(9), np.ones(1), 0.0)
    inputs = Inputs()
    inputs.select("owac")
    stream = StringIO()
    robot = hw.Equipment(
        SimpleNamespace(create_subscription=lambda *a: None),
        inputs,
        None,
        None,
        "test-only",
        SimpleNamespace(lower=np.zeros(8), upper=np.r_[np.ones(7), 0.1]),
        20,
        run_chunks=True,
        execution_horizons={"owac": 4},
        postprocessors={"owac": proc},
        physical_gripper_width=True,
        command_log=stream,
    )
    robot.armed = True
    robot.selected = hw.POLICY
    robot.health = lambda _: ""
    issued = []
    robot.publish = lambda target: issued.append(target.copy())
    monkeypatch.setattr(
        hw, "time", SimpleNamespace(monotonic=lambda: now, time_ns=lambda: round(now * 1e9))
    )
    acts = np.tile(np.r_[np.full(7, 0.3), -0.01], (12, 1))
    for i, width in enumerate((cfg["open_width_m"], 0.002, 0.002)):
        now = 10 + i / 10
        wall = round(now * 1e9)
        fb = Feedback(wall, wall, np.r_[np.zeros(7), width], "ROS")
        inputs.feedback(fb)
        obs = SimpleNamespace(at=now, capture=i + 1, feedback=fb)
        inputs.observation = obs
        inputs.prediction = Prediction("owac", obs, acts, 8, np.ones(1))
        robot.step(now, continuous=False)
    data = json.loads(stream.getvalue().splitlines()[-1])
    assert robot.sent == 3 and data["postprocess_phase"] == "opening"
    assert data["model_rad_m"][7] == -0.01 and issued[-1][7] == cfg["open_width_m"]
    assert np.any(issued[-1][:7] != acts[0, :7])
