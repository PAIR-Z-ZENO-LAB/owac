import json
from threading import Event, Thread
from types import SimpleNamespace

import numpy as np
import pytest

from owac.runtime.fruit_live import Feedback, Inputs, Prediction, Travel, command_row, row_at

PACKED_FRAME = bytes(480 * 1024)


def test_motion_history_cannot_carry_previous_reset_motion_into_a_new_run():
    inputs = Inputs()
    old, current = (SimpleNamespace(at=t) for t in (10.0, 10.2))
    inputs.history.extend((old, current))
    temporal = SimpleNamespace(cfg={"model": {"state_history_lags": [1, 3]}})
    inputs.begin_motion(10.1)
    assert inputs.history_for(temporal, current) == (current,)
    assert inputs.history_for(object(), current) == (old, current)


def test_checkpoint_list_and_menu_work_offline(tmp_path, monkeypatch, capsys):
    import curses
    import sys

    from owac.runtime import fruit_tui
    from owac.runtime.fruit_weights import catalog

    folder = tmp_path / "candidates"
    folder.mkdir()
    (folder / "catalog.json").write_text(
        json.dumps({"candidates": [dict(id="pose", label="香蕉对准", step=400)]})
    )
    monkeypatch.setenv("OWAC_FRUIT_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["fruit-infer", "--list"])
    assert fruit_tui.main() == 0
    assert "香蕉对准" in capsys.readouterr().out
    choices = catalog(tmp_path)
    keys = iter([curses.KEY_UP, 10])
    screen = SimpleNamespace(
        getmaxyx=lambda: (24, 80),
        getch=lambda: next(keys),
        timeout=lambda _: None,
        nodelay=lambda _: None,
    )
    monkeypatch.setattr(fruit_tui, "draw", lambda screen, lines: None)
    assert fruit_tui.choose_weight(screen, choices, "best") == "pose"


def test_only_released_policy_is_listed_and_selected_in_each_current_mode(
    tmp_path, monkeypatch, capsys
):
    import sys

    from owac.runtime import fruit_tui
    from owac.runtime.fruit_weights import catalog, default_policy

    folder = tmp_path / "candidates"
    folder.mkdir()
    (folder / "catalog.json").write_text(
        json.dumps(
            {
                "include_originals": False,
                "candidates": [dict(id="fruit-final", label="Fruit Final", step=800)],
            }
        )
    )
    monkeypatch.setenv("OWAC_FRUIT_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["fruit-infer", "--list"])
    assert fruit_tui.main() == 0
    assert capsys.readouterr().out.split() == ["fruit-final", "Fruit", "Final"]
    choices = catalog(tmp_path)
    assert list(choices) == ["fruit-final"]
    assert choices["fruit-final"].checkpoint == folder / "fruit-final" / "policy.pt"
    for check, grasp, place in (
        (False, None, False),
        (True, None, False),
        (False, None, True),
        (False, "direct", False),
    ):
        assert default_policy(choices, check=check, grasp=grasp, place=place) == "fruit-final"
    monkeypatch.setattr(sys, "argv", ["fruit-infer", "--grasp-test", "original"])
    with pytest.raises(SystemExit) as error:
        fruit_tui.main()
    assert error.value.code == 2


def image(camera, request, wall, *, board=None):
    sec, nanosec = divmod(wall, 10**9)
    return SimpleNamespace(
        camera_id=camera,
        request_sent_monotonic_ns=request,
        capture_done_monotonic_ns=request + 3,
        received_monotonic_ns=request + 1 + camera,
        fourcc="BG10",
        width=640,
        height=480,
        stride=1024,
        data=PACKED_FRAME,
        board_timestamp_us=board if board is not None else request,
        header=SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nanosec)),
    )


def test_released_fruit_and_flow_baselines_share_a_catalog(tmp_path):
    from owac.runtime.fruit_weights import catalog, default_policy

    folder = tmp_path / "candidates"
    folder.mkdir()
    (folder / "catalog.json").write_text(
        json.dumps(
            {
                "include_originals": False,
                "candidates": [
                    {"id": "fruit-final", "step": 800, "label": "Fruit Final"},
                    {
                        "id": "rawvla-pi05",
                        "step": 512,
                        "label": "RawVLA + π0.5",
                        "backend": "rawvla",
                        "assets": "baselines",
                    },
                ],
            }
        )
    )
    choices = catalog(tmp_path)
    assert choices["fruit-final"].rtc == folder / "fruit-final" / "rtc.json"
    assert choices["rawvla-pi05"].rtc is None
    assert choices["rawvla-pi05"].assets == tmp_path / "baselines"
    assert default_policy(choices, check=False, grasp=None, place=True) == "fruit-final"
    assert default_policy(choices, check=True, grasp=None, place=False) is None


def test_bowl_console_selects_the_trained_checkpoints_without_fruit_calibration(
    tmp_path, monkeypatch, capsys
):
    import sys
    from pathlib import Path

    from owac.runtime import fruit_tui
    from owac.runtime.fruit_weights import catalog, default_policy

    manifest = Path(__file__).resolve().parents[1] / "configs/runtime/bowl_catalog.json"
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    (candidates / "catalog.json").write_bytes(manifest.read_bytes())
    choices = catalog(tmp_path)
    assert default_policy(choices, check=False, grasp=None, place=False) == "owac"
    runs = tmp_path / "checkpoints/20261123/runs"
    assert choices["owac"].checkpoint == (
        tmp_path / "checkpoints/20261124-local/motion-history/policy-best.pt"
    )
    assert choices["owac"].execution_horizon == 4
    assert choices["owac"].postprocess == (
        tmp_path / "checkpoints/20261124-local/complete-flow/flow-policy.json"
    )
    for key, backend, step in (
        ("owac", "owac", 9500),
        ("dino-corp", "dino-corp", 9000),
        ("rawvla-pi05", "rawvla", 10000),
        ("rawild-pi05", "rawild", 10000),
    ):
        choice = choices[key]
        assert (choice.backend, choice.step, choice.rtc) == (backend, step, None)
        if backend != "owac":
            assert choice.checkpoint == runs / backend / "best.pt"
    monkeypatch.setenv("OWAC_FRUIT_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["bowl-infer", "--list"])
    assert fruit_tui.main() == 0
    assert [line.split()[0] for line in capsys.readouterr().out.splitlines()] == list(choices)
    monkeypatch.setattr(sys, "argv", ["bowl-infer", "--place-test"])
    with pytest.raises(SystemExit) as error:
        fruit_tui.main()
    assert error.value.code == 2


def test_slow_policy_uses_camera_history_instead_of_previous_inference_times():
    inputs = Inputs()
    started, release = Event(), Event()
    seen = []

    class SlowPolicy:
        def predict_live(self, history, unpack):
            seen.append([obs.capture for obs in history])
            if len(seen) == 1:
                started.set()
                assert release.wait(2)
            return np.zeros((12, 8))

    worker = Thread(target=inputs.infer, args=({"rawvla": SlowPolicy()}, None, None))
    inputs.feedback(Feedback(999, 999, np.zeros(8), "CAN"))
    inputs.select("rawvla")
    worker.start()
    try:
        for req in range(1, 36):
            for camera in (0, 1):
                inputs.frame(camera, image(camera, req, 1000))
            if req == 1:
                assert started.wait(2)
        release.set()
        with inputs.condition:
            assert inputs.condition.wait_for(
                lambda: (
                    inputs.prediction is not None and inputs.prediction.observation.capture == 35
                ),
                2,
            )
        assert seen == [[1], list(range(5, 36))]
        inputs.select("another")
        assert [obs.capture for obs in inputs.history] == [35]
    finally:
        release.set()
        with inputs.condition:
            inputs.stopping = True
            inputs.condition.notify_all()
        worker.join(3)


def test_pair_and_causal_feedback():
    inputs = Inputs()
    wall = 10**9
    inputs.feedback(Feedback(wall - 2, wall - 2, np.ones(8), "CAN"))
    inputs.feedback(Feedback(wall - 4, wall - 3, np.ones(8), "ROS"))
    inputs.feedback(Feedback(wall - 9, wall - 1, np.ones(8), "ROS"))
    inputs.feedback(Feedback(wall - 1, wall + 5, np.ones(8), "ROS"))
    inputs.frame(0, image(0, 1, wall))
    inputs.frame(1, image(1, 2, wall))
    assert inputs.observation is None
    inputs.frame(0, image(0, 2, wall))
    paired = inputs.observation
    assert paired.capture == 2 and paired.feedback.stamp == wall - 4
    assert [msg.camera_id for msg in paired.frames] == [0, 1]
    inputs.frame(0, image(0, 3, wall, board=2))
    inputs.frame(1, image(1, 3, wall))
    assert inputs.observation is paired
    assert inputs.at("ROS", wall + 10**9) is None
    p = Prediction("best", paired, np.zeros((12, 8)), 10)
    assert row_at(p, paired.at - 1) is None
    assert row_at(p, paired.at + 0.069) == 2
    assert row_at(p, paired.at + 11 / 30) == 11
    assert row_at(p, paired.at + 0.40) is None


def test_switch_discards_in_flight_old_checkpoint():
    inputs = Inputs()
    started, release = Event(), Event()
    inputs.feedback(Feedback(999, 999, np.zeros(8), "CAN"))

    def predict(model, raw, state):
        if model == "best":
            started.set()
            assert release.wait(2)
        return np.full((1, 12, 8), 1 if model == "best" else 2)

    worker = Thread(
        target=inputs.infer,
        args=({"best": "best", "final": "final"}, predict, lambda *args: np.zeros((1, 1))),
    )
    worker.start()
    try:
        inputs.select("best")
        for view in (0, 1):
            inputs.frame(view, image(view, 10, 1000))
        assert started.wait(2)
        inputs.select("final")
        assert inputs.prediction is None
        release.set()
        with inputs.condition:
            assert inputs.condition.wait_for(lambda: inputs.prediction is not None, 2)
            assert inputs.prediction.checkpoint == "final"
            np.testing.assert_array_equal(inputs.prediction.actions, np.full((12, 8), 2))
    finally:
        release.set()
        with inputs.condition:
            inputs.stopping = True
            inputs.condition.notify_all()
        worker.join(2)
    assert not worker.is_alive()


def test_command_row_reports_which_input_cannot_be_used():
    obs = SimpleNamespace(at=10.0, feedback=SimpleNamespace(source="ROS"))
    prediction = Prediction("best", obs, np.zeros((12, 8)), 10)
    assert command_row(prediction, "best", 10.070) == 2
    with pytest.raises(ValueError, match="尚未收到"):
        command_row(None, "best", 10.070)
    with pytest.raises(ValueError, match="当前权重为 final"):
        command_row(prediction, "final", 10.070)
    with pytest.raises(ValueError, match="晚 1.0 ms"):
        command_row(prediction, "best", 9.999)
    with pytest.raises(ValueError, match="201.0 ms，上限 200 ms"):
        command_row(prediction, "best", 10.201)
    obs.feedback.source = "CAN"
    with pytest.raises(ValueError, match="CAN 反馈"):
        command_row(prediction, "best", 10.070)


def test_direct_uses_the_trained_window_while_new_camera_frames_arrive():
    obs = SimpleNamespace(at=10.0, feedback=SimpleNamespace(source="ROS"))
    p = Prediction("best", obs, np.zeros((12, 8)), 90)
    for age, expected in ((0.2033, 6), (0.2158, 6), (11 / 30, 11)):
        now = obs.at + age
        latest = SimpleNamespace(at=now - 0.010)
        assert command_row(p, "best", now, direct=True, latest_observation=latest) == expected
    with pytest.raises(ValueError, match="最新配对画面"):
        command_row(p, "best", 10.216, direct=True, latest_observation=obs)
    with pytest.raises(ValueError, match="上限"):
        command_row(p, "best", 10.370, direct=True, latest_observation=SimpleNamespace(at=10.36))
    obs.feedback.source = "CAN"
    with pytest.raises(ValueError, match="CAN 反馈"):
        command_row(p, "best", 10.010, direct=True, latest_observation=obs)


def test_camera_format_and_inconsistent_pair():
    inputs = Inputs()
    camera = image(1, 10, 1000)
    camera.fourcc = "UYVY"
    with pytest.raises(ValueError, match="BG10"):
        inputs.frame(1, camera)
    inputs.frame(0, image(0, 10, 1000))
    camera.fourcc = "BG10"
    camera.capture_done_monotonic_ns += 1
    with pytest.raises(ValueError, match="完成时间"):
        inputs.frame(1, camera)


@pytest.fixture
def limits(tmp_path):
    joint = '<joint name="joint{}"><limit lower="-2" upper="2" velocity="5"/></joint>'
    (tmp_path / "nero_description.urdf").write_text(
        "<robot>" + "".join(joint.format(i) for i in range(1, 8)) + "</robot>"
    )
    (tmp_path / "nero_with_gripper_description.xacro").write_text(
        '<robot><joint name="gripper_joint1"><limit lower="0" upper=".05" velocity="3"/></joint>'
        '<joint name="gripper_joint2"><limit lower="-.05" upper="0" velocity="3"/></joint></robot>'
    )
    return Travel(tmp_path, 0.2)


def test_limits_use_measured_pose_and_gripper_travel(limits):
    measured = np.r_[np.full(7, 1.995), 0.099]
    prediction = np.full(8, 20.0)
    out = limits.command(prediction, measured)
    np.testing.assert_allclose(out, [2] * 7 + [0.1])
    out = limits.command(-prediction, measured)
    np.testing.assert_allclose(out - measured, [-0.03] * 7 + [-0.003])
    assert limits.command(prediction, measured)[7] == pytest.approx(0.1)


def test_reset_pose_and_coordinated_commands(tmp_path, limits):
    profile = tmp_path / "reset.json"
    profile.write_text(json.dumps({"joints_deg": [0, 30, -15, 75, 10, -8, 90], "gripper_mm": 6}))
    target = limits.load_reset_pose(profile)
    np.testing.assert_allclose(target, [*np.deg2rad([0, 30, -15, 75, 10, -8, 90]), 0.006])
    current = np.array([0.1] * 7 + [0.09])
    first, final = limits.reset_command(target, current)
    assert not final
    np.testing.assert_allclose(
        (first - current) / (target - current), 1 / limits.reset_ticks(target, current)
    )
    for _ in range(2 + int(limits.reset_ticks(target, current))):
        command, final = limits.reset_command(target, current)
        np.testing.assert_array_less(np.abs(command - current), limits.step + 1e-14)
        current = command
        if final:
            break
    assert final
    np.testing.assert_array_equal(current, target)
    current[0] = limits.upper[0] + limits.step[0] / 2
    assert limits.reset_command(target, current)[0][0] <= limits.upper[0]
    current[0] = limits.upper[0] + 2 * limits.step[0]
    with pytest.raises(ValueError, match="有效范围"):
        limits.reset_command(target, current)


@pytest.mark.parametrize("angles,width", [([0] * 6, 0), ([0] * 7, 101), ([float("nan")] * 7, 0)])
def test_invalid_reset_profile(tmp_path, limits, angles, width):
    profile = tmp_path / "reset.json"
    profile.write_text(json.dumps({"joints_deg": angles, "gripper_mm": width}))
    with pytest.raises(ValueError):
        limits.load_reset_pose(profile)
