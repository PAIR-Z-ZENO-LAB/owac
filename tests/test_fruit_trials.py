import json
from types import SimpleNamespace

import numpy as np
import pytest

from owac.runtime import fruit_tui
from owac.runtime.fruit_live import Inputs
from owac.runtime.fruit_trials import GraspTrials


def results(trials):
    return [r for r in map(json.loads, trials.path.read_text().splitlines()) if r["event"] == "end"]


def test_grasps_preserve_the_verdict_and_partially_completed_trials(tmp_path):
    trials = GraspTrials(tmp_path)
    trials.begin("a", 12, 71)
    trials.begin("a", 13, 73)
    q = np.linspace(0, 0.4, 8)
    fb = SimpleNamespace(stamp=9, values=q)
    trials.observe(13, 73, fb, q + 0.03, q + 0.02, (10, q + 0.01))
    trials.finish("lifted", 14, 75)
    trials.finish("missed", 15, 75)
    trials.begin("a", 16, 75)
    trials.finish("missed", 17, 75)
    trials.begin("b", 20, 75)
    trials.finish("missed", 26.25, 102)
    trials.begin("a", 27, 102)
    trials.finish("unscored", 29, 104)
    verdicts = results(trials)
    assert [r["outcome"] for r in verdicts] == ["lifted", "unscored", "missed", "unscored"]
    assert [r["commands"] for r in verdicts] == [4, 0, 27, 2]
    assert verdicts[2]["elapsed_s"] == 6.25
    [sample] = verdicts[0]["samples"]
    assert sample["commands"] == 2 and sample["mux_stamp_ns"] == 10
    np.testing.assert_allclose(sample["mux_rad_m"], q + 0.01)
    np.testing.assert_allclose(sample["feedback_rad_m"], q)
    np.testing.assert_allclose(sample["model_rad_m"], q + 0.03)
    assert verdicts[1]["samples"] == []
    assert "1/1" in trials.status("a", 30) and "0/1" in trials.status("b", 30)


@pytest.mark.parametrize(("preset", "success"), [("direct", "lifted"), ("placement", "placed")])
def test_panel_counts_an_operator_verdict_after_pause_and_ignores_duplicate_keys(
    tmp_path, monkeypatch, preset, success
):
    trials = GraspTrials(tmp_path, preset)
    choices = {"best": SimpleNamespace(label="原版")}
    inputs = Inputs()
    inputs.select("best")
    robot = SimpleNamespace(
        running=False,
        armed=False,
        engaging=0,
        reset_deadline=None,
        direct=True,
        run_chunks=False,
        selected="",
        can="offline",
        last_command=None,
        last_mux=None,
        last_model_target=None,
        last_policy_target=None,
        reset_pose=None,
        info="",
        sent=0,
        output_count=0,
        tick=lambda now: None,
    )
    robot.connect = lambda: setattr(robot, "armed", True)

    def step(now, continuous):
        if robot.armed:
            robot.running = continuous
            robot.sent += 1

    robot.step = step
    robot.pause = lambda reason: setattr(robot, "running", False)
    frames = ["r", "f", "c", "r", " ", "s", "ss", "r", "f", "q"]
    keys = iter([k for frame in frames for k in [*(ord(c) for c in frame), -1]])
    screen = SimpleNamespace(nodelay=lambda value: None, getch=lambda: next(keys))
    renders = []
    monkeypatch.setattr(fruit_tui, "draw", lambda screen, lines: renders.append(str(lines)))
    monkeypatch.setattr(fruit_tui.time, "sleep", lambda _: None)
    fruit_tui.panel(screen, inputs, robot, 20, choices, trials)
    assert robot.sent == 2 and not robot.running
    assert [r["outcome"] for r in results(trials)] == [success, "missed"]
    assert "1/2（50%）" in renders[-1]
