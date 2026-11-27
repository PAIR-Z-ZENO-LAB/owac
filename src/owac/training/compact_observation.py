"""Explicit RAW/view and robot-state contracts for compact manipulation policies."""

import numpy as np

LIBERO_OBSERVATION = {
    "domain": "libero",
    "views": {"agentview": "agentview_raw_uint8", "eye_in_hand": "wrist_raw_uint8"},
    "raw_scale": 255.0,
    "state": "libero_joint",
    "task_field": "task_description",
}


def observation_spec(cfg):
    return cfg.get("observation", LIBERO_OBSERVATION)


def raw_input_scale(cfg):
    """Native DN calibration belongs to the encoder, not the dataset's RGB scale."""
    sensor = (
        cfg.get("model", {}).get("raw_encoder", {}).get("measurement_encoder", {}).get("sensor")
    )
    if sensor and sensor["units"] == "dn":
        return 1.0
    return observation_spec(cfg)["raw_scale"]


def axisangle_matrix(values):
    """Rodrigues formula with analytic limits at zero, without a cutoff angle."""
    values = np.asarray(values, dtype=np.float32)
    x, y, z = np.moveaxis(values, -1, 0)
    zero = np.zeros_like(x)
    skew = np.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*values.shape[:-1], 3, 3)
    angle = np.linalg.norm(values, axis=-1)
    sine = np.sinc(angle / np.pi)[..., None, None]
    cosine = (0.5 * np.sinc(angle / (2 * np.pi)) ** 2)[..., None, None]
    return np.eye(3, dtype=np.float32) + sine * skew + cosine * (skew @ skew)


def quaternion_matrix(values, *, order):
    """Quaternion to rotation matrix; order belongs to the platform contract."""
    q = np.asarray(values, dtype=np.float32)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    if order == "xyzw":
        x, y, z, w = np.moveaxis(q, -1, 0)
    elif order == "wxyz":
        w, x, y, z = np.moveaxis(q, -1, 0)
    else:
        raise ValueError(f"Unsupported quaternion order: {order}")
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        -1,
    ).reshape(*q.shape[:-1], 3, 3)


def rotation_columns(matrix):
    """First two rotation-matrix columns, in the same order in data and inference."""
    return np.concatenate((matrix[..., :, 0], matrix[..., :, 1]), -1)


def archive_state(archive, cfg):
    state = observation_spec(cfg)["state"]
    if state == "libero_joint":
        return archive["states"][:, 1:10].astype(np.float32)
    if state == "libero_joint_eef":
        joints = archive["states"][:, 1:10].astype(np.float32)
        pose = archive["robot_states"][:, :6].astype(np.float32)
        rotation = rotation_columns(axisangle_matrix(pose[:, 3:6]))
        return np.concatenate((joints, pose[:, :3], rotation), -1)
    if state == "robotwin_eef":
        return robotwin_pose_state(archive["robot_states"])
    if state == "robotwin_joint_eef":
        # "actions" stores current drive targets / gripper commands, not measured
        # joint qpos. The native observation exposes the same fields; predict the
        # next recorded target sequence while endpose supplies physical feedback.
        return np.concatenate(
            (archive["actions"].astype(np.float32), robotwin_pose_state(archive["robot_states"])),
            -1,
        )
    raise ValueError(f"Unsupported compact state contract: {state}")


def libero_state(observation, cfg):
    joints = np.concatenate(
        (observation["robot0_joint_pos"], observation["robot0_gripper_qpos"])
    ).astype(np.float32)
    state = observation_spec(cfg)["state"]
    if state == "libero_joint":
        return joints
    if state == "libero_joint_eef":
        rotation = rotation_columns(quaternion_matrix(observation["robot0_eef_quat"], order="xyzw"))
        return np.concatenate((joints, observation["robot0_eef_pos"], rotation)).astype(np.float32)
    raise ValueError(f"Unsupported LIBERO compact state contract: {state}")


def robotwin_pose_state(values):
    """Native RoboTwin endpose: left/right xyz, wxyz quaternion, gripper."""
    values = np.asarray(values, dtype=np.float32)
    pose = values.reshape(*values.shape[:-1], 2, 8)
    rotation = rotation_columns(quaternion_matrix(pose[..., 3:7], order="wxyz"))
    return np.concatenate((pose[..., :3], rotation, pose[..., 7:8]), -1).reshape(
        *values.shape[:-1], 20
    )


def robotwin_state(observation, cfg):
    state = observation_spec(cfg)["state"]
    if state not in ("robotwin_eef", "robotwin_joint_eef"):
        raise ValueError(f"Unsupported RoboTwin compact state contract: {state}")
    pose = observation["endpose"]
    values = np.concatenate(
        (
            pose["left_endpose"],
            [pose["left_gripper"]],
            pose["right_endpose"],
            [pose["right_gripper"]],
        )
    )
    geometry = robotwin_pose_state(values)
    if state == "robotwin_joint_eef":
        return np.concatenate((observation["joint_action"]["vector"], geometry)).astype(np.float32)
    return geometry


def task_text(metadata, cfg):
    field = observation_spec(cfg)["task_field"]
    text = metadata[field]
    return text.replace("_", " ") if field == "task_name" else text
