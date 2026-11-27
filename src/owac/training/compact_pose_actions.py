"""Current-reference SE(3) chunks and native RoboTwin end-pose conversion.

Each arm uses xyz, the first two rotation-matrix columns, and an absolute
gripper command. Relative chunks share the current observation's reference;
world-space chunks can be combined across observations before native execution.
"""

import numpy as np

from owac.training.compact_observation import robotwin_pose_state, rotation_columns


def columns_matrix(values):
    """Continuous 6D Gram-Schmidt map; zero/collinear columns are undefined."""
    values = np.asarray(values, dtype=np.float64)
    first, second = values[..., :3], values[..., 3:6]
    with np.errstate(divide="raise", invalid="raise"):
        first = first / np.linalg.norm(first, axis=-1, keepdims=True)
        second = second - (first * second).sum(-1, keepdims=True) * first
        second = second / np.linalg.norm(second, axis=-1, keepdims=True)
    return np.stack((first, second, np.cross(first, second)), axis=-1)


def archive_endpose_actions(archive):
    """Use recorded actual end poses and original commanded grippers, not FK."""
    commands = archive["actions"]
    values = np.concatenate(
        (
            archive["left_endpose"],
            commands[:, 6:7],
            archive["right_endpose"],
            commands[:, 13:14],
        ),
        axis=-1,
    )
    return robotwin_pose_state(values)


def transform_endpose_actions(actions, state, *, to_world):
    """Convert (..., horizon, 20) using one current (..., state_dim) reference."""
    values = np.asarray(actions)
    target = values.reshape(*values.shape[:-1], 2, 10)
    state = np.asarray(state)[..., -20:]
    reference = state.reshape(*state.shape[:-1], 2, 10)[..., None, :, :]
    current_rotation = columns_matrix(reference[..., 3:9])
    target_rotation = columns_matrix(target[..., 3:9])
    if to_world:
        position = reference[..., :3] + (current_rotation @ target[..., :3, None])[..., 0]
        rotation = current_rotation @ target_rotation
    else:
        inverse = np.swapaxes(current_rotation, -1, -2)
        position = (inverse @ (target[..., :3] - reference[..., :3])[..., None])[..., 0]
        rotation = inverse @ target_rotation
    return (
        np.concatenate((position, rotation_columns(rotation), target[..., 9:10]), -1)
        .reshape(values.shape)
        .astype(np.float32)
    )


def native_endpose_actions(world_actions):
    """Convert world-space 20D targets to native xyz/wxyz/gripper after aggregation."""
    from scipy.spatial.transform import Rotation

    values = np.asarray(world_actions)
    pose = values.reshape(*values.shape[:-1], 2, 10)
    matrix = columns_matrix(pose[..., 3:9])
    xyzw = Rotation.from_matrix(matrix.reshape(-1, 3, 3)).as_quat()
    wxyz = xyzw[:, [3, 0, 1, 2]].reshape(*pose.shape[:-1], 4)
    return (
        np.concatenate((pose[..., :3], wxyz, pose[..., 9:10]), -1)
        .reshape(*values.shape[:-1], 16)
        .astype(np.float32)
    )
