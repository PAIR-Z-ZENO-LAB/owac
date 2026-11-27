"""Recorded CPU predictions and an isolated NERO delivery check; no ROS graph or CAN device."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


def recorded(root, samples):
    import torch

    from owac.frameworks.raw_compact.factory import make_policy
    from owac.runtime.fruit_weights import catalog
    from owac.training.teleop_policy_eval import predict_actions

    torch.set_num_threads(3)
    torch.manual_seed(20261123)
    choice = catalog(root)["owac"]
    weights = torch.load(choice.checkpoint, weights_only=False, map_location="cpu", mmap=True)
    if choice.step != weights["step"] or choice.step != 6000:
        raise ValueError("Wrong pouring OWAC checkpoint")
    model = make_policy(weights["config"]).eval()
    model.load_state_dict(weights["model"], strict=True)
    with np.load(samples) as inputs:
        raw, states = inputs["raw"], inputs["state"]
        results = np.concatenate(
            [
                predict_actions(model, raw[start : start + 2], states[start : start + 2])
                for start in range(0, len(raw), 2)
            ]
        )
        if results.shape != (30, 12, 8) or not np.isfinite(results).all():
            raise ValueError("Malformed predictions")
        return dict(
            predictions=results.tolist(),
            states=states.tolist(),
            episodes=inputs["episode"].tolist(),
        )


def isolate(root, original, candidate, current, log):
    from builtin_interfaces.msg import Time
    from piper_nero_teleop import nero_ctrl_node as driver

    from owac.runtime import fruit_robot
    from owac.runtime.fruit_live import Feedback, Inputs, Prediction, Travel

    raw = np.array(candidate["predictions"])
    np.testing.assert_array_equal(raw, original["predictions"])
    urdf = Path.home() / "owac/agilx-teleop-collection/agx_arm_urdf/nero/urdf"
    travel = Travel(urdf, 0.2)
    goal = travel.load_reset_pose(root / "reset-pose.json")
    source_pose = current.copy()
    inputs = Inputs()
    inputs.select("owac")
    clock = SimpleNamespace(now=lambda: SimpleNamespace(to_msg=Time))
    node = SimpleNamespace(create_subscription=lambda *args: None, get_clock=lambda: clock)
    native = driver.NeroCtrlNode.__new__(driver.NeroCtrlNode)
    native.gripper_cmd_limit_enable = True
    native.gripper_cmd_min_raw, native.gripper_cmd_max_raw, native.gripper_cmd_deadband_raw = (
        0,
        100000,
        200,
    )
    issued = []
    robot = fruit_robot.Equipment(
        node,
        inputs,
        None,
        None,
        "unused",
        travel,
        20,
        reset_pose=goal,
        run_chunks=True,
        physical_gripper_width=True,
        command_log=log,
    )
    robot.command_pub = SimpleNamespace(publish=lambda msg: issued.append(np.array(msg.position)))
    robot.health = lambda _: ""
    robot.armed, robot.selected = True, fruit_robot.POLICY
    now = 100.0
    timer = SimpleNamespace(monotonic=lambda: now, time_ns=lambda: int(now * 1e9))
    with patch.object(fruit_robot, "time", timer):

        def feedback(values):
            frame = Feedback(timer.time_ns(), timer.time_ns(), values.copy(), "ROS")
            inputs.feedback(frame)
            return frame

        feedback(current)
        robot.start_reset(now)
        assert robot.reset_deadline is not None
        for _ in range(1800):
            if robot.reset_deadline is None:
                break
            now += 1 / 30
            feedback(current)
            before = len(issued)
            robot.tick(now)
            if len(issued) > before:
                assert np.all(np.abs(issued[-1] - current) <= travel.step + 1e-8)
                assert np.all((travel.lower <= issued[-1]) & (issued[-1] <= travel.upper))
                current = issued[-1].copy()
        assert "复位完成" in robot.info and robot.reset_deadline is None and not robot.running
        np.testing.assert_array_equal(current, goal)
        reset_waypoints = len(issued)
        reflected = negative = 0
        for index, (plan, state) in enumerate(zip(raw, candidate["states"], strict=True)):
            obs = SimpleNamespace(capture=index + 1, feedback=feedback(np.asarray(state)))
            prediction = Prediction("owac", obs, plan, 0)
            for row, action in enumerate(plan):
                now += 1 / 30
                feedback(np.asarray(state))
                robot.last_mux = (timer.time_ns(), issued[-1].copy())
                robot.publish_policy(action, prediction, row)
                target = issued[-1]
                np.testing.assert_array_equal(target[:7], action[:7])
                assert target[7] == np.clip(action[7], 0, travel.upper[7])
                if action[7] < 0:
                    negative += 1
                    reflected += native._sanitize_gripper_raw_cmd(round(action[7] * 1e6)) > 0
                    assert native._sanitize_gripper_raw_cmd(round(target[7] * 1e6)) == 0
    return dict(
        independent_episodes=len(set(candidate["episodes"])),
        paired_frames=len(raw),
        action_rows=len(raw) * 12,
        identical_model_outputs=True,
        unchanged_seven_joints=True,
        negative_widths=negative,
        formerly_reflected_widths=int(reflected),
        reset_source_rad_m=source_pose.tolist(),
        reset_goal_rad_m=goal.tolist(),
        reset_waypoints=reset_waypoints,
        physical_motor_commands=0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("recorded", "isolated"))
    parser.add_argument("root", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    work = args.work
    if args.mode == "recorded":
        result = recorded(args.root, work / "heldout-probes.npz")
    else:
        records = [
            json.loads(line)
            for line in (work / "live-owac.log").read_text().splitlines()
            if line.startswith('{"policy"')
        ]
        assert len(records) == 1 and records[0]["policy_commands"] == 0
        with (work / "isolated-commands.jsonl").open("w", buffering=1) as log:
            result = isolate(
                args.root,
                json.loads((work / "original.json").read_text()),
                json.loads((work / "candidate.json").read_text()),
                np.array(records[0]["measured_rad_m"]),
                log,
            )
        commands = [
            json.loads(line) for line in (work / "isolated-commands.jsonl").read_text().splitlines()
        ]
        assert len(commands) == result["action_rows"] and all(
            row["feedback_ns"] and row["latest_mux_ns"] for row in commands
        )
    args.output.write_text(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
