import json
from pathlib import Path

import numpy as np

from owac.runtime.bowl_postprocess import BowlPostprocessor


def profile():
    name = Path(__file__).resolve().parents[1] / "tests/fixtures/bowl.json"
    return json.loads(name.read_text())


def test_release_requires_a_grasp_and_persists_until_next_approach():
    cfg = profile()
    for key, weights in (("release", [1.0, 0.0]), ("next_grasp", [0.0, 1.0])):
        cfg["events"][key] = dict(motion=[0.0] * 9, scene=weights, bias=-0.5)
    proc = BowlPostprocessor(cfg)
    opening = cfg["open_width_m"]
    q = np.zeros(8)
    target = np.zeros((4, 8))
    now = 0.0

    def apply(width, scene):
        nonlocal now
        now += 1 / 30
        q[7] = width
        return proc.command(target, 0, q, np.array(scene), now)[7]

    assert apply(0, [1, 0]) == 0
    apply(opening, [0, 0])
    apply(0, [0, 0])
    assert proc.phase == "holding"
    assert apply(0, [0, 0]) == 0
    assert apply(0, [1, 0]) == opening
    assert apply(0, [0, 0]) == opening
    assert apply(opening, [0, 0]) == opening
    assert apply(opening, [0, 0]) == opening
    assert apply(opening, [0, 1]) == 0
    assert proc.phase == "native"
    proc.reset()
    assert apply(0, [1, 0]) == 0


def test_smoothing_starts_at_measured_pose_and_leaves_native_aperture_independent():
    cfg = profile()
    p = BowlPostprocessor(cfg)
    measured = np.zeros(8)
    plan = np.tile(np.r_[np.full(7, 0.3), 0.072], (4, 1))
    a = p.command(plan, 0, measured, None, 10.0)
    assert np.max(np.abs(a[:7])) < 0.3
    assert a[7] == 0.072
    p.arm.reset()
    measured[:7] = 0.1
    cmd = p.command(plan, 0, measured, None, 100.0)
    assert np.all(cmd[:7] > a[:7])
