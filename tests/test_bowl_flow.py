import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
from owac.runtime.bowl_flow import BowlFlow, pose_jacobian, translate_path
from owac.runtime.bowl_postprocess import hand_center


def geometry():
    path = Path(__file__).resolve().parents[1] / "tests/fixtures/bowl.json"
    profile = json.loads(path.read_text())
    return profile["geometry"], np.asarray(profile["joint_limits_rad"])


def test_shift_retains_wrist_orientation_and_corrects_both_planar_axes():
    geo, limits = geometry()
    start = np.array([-0.1, 0.31, -0.43, 1.6, 0.12, -0.24, 0.85])
    trajectory = np.stack((start, start + 0.03, start - 0.02))
    xyz, rot, jac = pose_jacobian(trajectory, geo)
    np.testing.assert_allclose(xyz, hand_center(trajectory, geo), atol=1.0e-8)
    for axis in range(7):
        change = np.eye(7)[axis] * 1.0e-6
        observed = (hand_center(trajectory + change, geo) - xyz) / 1.0e-6
        np.testing.assert_allclose(jac[:, :3, axis], observed, atol=4.0e-7)
    offsets = np.array([[0.0, 0.0, 0.0], [0.018, -0.013, 0.0], [-0.011, 0.023, 0.0]])
    shifted = translate_path(trajectory, offsets, geo, limits)
    pos, actual, _ = pose_jacobian(shifted, geo)
    np.testing.assert_allclose(pos, xyz + offsets, atol=1.0e-6)
    np.testing.assert_allclose(actual, rot, atol=1.0e-6)


def test_blocked_native_jaw_and_jitter_cannot_interrupt_a_contact():
    from collections import deque

    from owac.runtime.bowl_postprocess import JointPlanSmoother

    geo, limits = geometry()
    p = BowlFlow.__new__(BowlFlow)
    p.profile = dict(
        geometry=geo,
        joint_limits_rad=limits,
        layout_frames=3,
        stroke_window=4,
        command_step_rad=[0.04] * 7,
        width_split_m=0.03,
        closed_width_m=0.0,
        open_width_m=0.065,
        ack_width_m=[0.005, 0.045, 0.005, 0.045],
        contact_radius_m=[0.006] * 5,
        contact_angle_rad=[0.02] * 4,
        held_width_variation_m=0.0005,
    )
    p.contact_geometry = geo
    p.geo = geo
    p.limits = limits
    p.samples = deque(maxlen=3)
    p.arm = JointPlanSmoother(100, limits)
    p.reset()
    p.centers = np.zeros((4, 2))
    q = np.array([-0.1, 0.31, -0.43, 1.6, 0.12, -0.24, 0.85, 0.063])
    scene = np.zeros(2)
    plan = np.tile(q, (4, 1))
    p.bank = dict(native_span=np.ones((5, 7)), native_gain=np.ones(5), track_span=np.zeros((5, 7)))
    for stage in (0, 1, 2, 3):
        p.stage = stage
        p.phase = "approach"
        p.duration = 1.0
        p.elapsed = 1.0
        p.stroke = False
        p.path = np.tile(q[:7], (5, 1))
        p.goal = hand_center(q, geo)
        p.orientation = pose_jacobian(q, geo)[1][0]
        p.arm_plan = None
        plan[:, 7] = 0.1 if stage % 2 == 0 else 0.0
        q[7] = 0.065 if stage % 2 == 0 else 0.001
        first = p.command(plan, 0, q, scene, 100 + stage)
        assert first[7] == (0.0 if stage % 2 == 0 else 0.065)
        plan[:, :7] += 0.09
        second = p.command(plan, 1, q, scene, 100 + stage + 1 / 30)
        np.testing.assert_array_equal(second[:7], first[:7])
        assert p.stage == stage
        q[7] = 0.004 if stage % 2 == 0 else 0.048
        p.command(plan, 2, q, scene, 100 + stage + 2 / 30)
        assert p.stage == stage + 1
        p.recheck = None
    p.suspend()
    assert p.arm.last is None and p.stage == 4
    p.reset()
    assert p.phase == "locate_bowls" and p.stage == 0 and p.centers is None
