from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.experiments.prepare_pouring_field import reset_profile


def test_reset_uses_an_observed_medoid_not_the_mean_and_bounds_the_width():
    travel = SimpleNamespace(lower=np.zeros(8), upper=np.ones(8), step=np.ones(8) * 0.01)
    starts = {
        key: {"start": [value] * 7 + [-0.001], "observation_ns": index + 1}
        for index, (key, value) in enumerate([("far", 0.9), ("a", 0.1), ("b", 0.12)])
    }
    profile, proof = reset_profile(starts, travel)
    assert proof["episode"] == "b" and proof["clipped_axes"] == [7]
    np.testing.assert_allclose(np.deg2rad(profile["joints_deg"]), [0.12] * 7)
    assert 0 <= profile["gripper_mm"] < 1e-10
    starts["b"]["start"][0] = float("nan")
    with pytest.raises(ValueError):
        reset_profile(starts, travel)


def test_field_frame_trace_is_bounded_paired_and_shared_with_pause(tmp_path):
    pytest.importorskip("rclpy")
    from owac.runtime.fruit_live import Feedback, Inputs, Observation
    from owac.runtime.fruit_robot import Equipment

    state = Feedback(100, 100, np.zeros(8), "ROS")
    frames = tuple(
        SimpleNamespace(data=bytes([value] * 4), width=2, height=1, stride=4, sequence=value)
        for value in (1, 2)
    )
    with (tmp_path / "commands.jsonl").open("w") as log:
        robot = Equipment(
            SimpleNamespace(create_subscription=lambda *args: None),
            Inputs(),
            None,
            None,
            "not-a-bus",
            None,
            20,
            command_log=log,
            command_frame_interval_s=0.5,
        )
        first = Observation(100, 1.0, 101, frames, state)
        second = Observation(200, 1.1, 201, frames, state)
        saved = robot.trace_observation(first)
        assert robot.trace_observation(first, pause=True) == saved
        assert robot.trace_observation(second) is None
        robot.pending_frame_write.result()
        later = robot.trace_observation(second, pause=True)
        robot.finish_frame_writes()
    for name, wall in ((saved, 101), (later, 201)):
        with np.load(tmp_path / Path(name)) as data:
            assert int(data["wall_ns"]) == wall
            np.testing.assert_array_equal(data["u2_bg10"], [1] * 4)
            np.testing.assert_array_equal(data["u3_bg10"], [2] * 4)
