"""Explicit native, current-joint-offset, and current-reference end-pose labels."""

import numpy as np

from owac.training.compact_observation import observation_spec
from owac.training.compact_pose_actions import archive_endpose_actions, transform_endpose_actions


def action_representation(cfg):
    """Explicit label/inference coordinates, with unchanged native defaults."""
    spec = observation_spec(cfg)
    representation = spec.get("action_representation", "native")
    if representation == "robotwin_relative_eef":
        if (
            spec["domain"] != "robotwin2"
            or spec["state"] not in ("robotwin_eef", "robotwin_joint_eef")
            or cfg["model"]["action_dim"] != 20
            or spec.get("relative_action_mask") is not None
        ):
            raise ValueError(
                "Relative RoboTwin end poses require 20D actions and current EEF state"
            )
    elif representation != "native":
        raise ValueError(f"Unknown action representation: {representation}")
    return representation


def archive_actions(archive, cfg):
    if action_representation(cfg) == "robotwin_relative_eef":
        return archive_endpose_actions(archive)
    return archive["actions"][:, : cfg["model"]["action_dim"]].astype(np.float32)


def encode_actions(actions, state, mask, representation="native"):
    if representation == "robotwin_relative_eef":
        return transform_endpose_actions(actions, state, to_world=False)
    return relative_actions(actions, state, mask)


def decode_actions(actions, state, cfg):
    if action_representation(cfg) == "robotwin_relative_eef":
        return transform_endpose_actions(actions, state, to_world=True)
    return absolute_actions(actions, state, observation_spec(cfg).get("relative_action_mask"))


def relative_actions(actions, state, mask):
    """Reference every prediction in a chunk to the same observed state."""
    if mask is None:
        return actions
    values = np.asarray(actions).copy()
    width = len(mask)
    reference = np.where(mask, np.asarray(state)[..., :width], 0)
    values[..., :width] -= np.expand_dims(reference, -2)
    return values


def absolute_actions(actions, state, mask):
    """Restore native targets before execution, including multi-turn joint values."""
    if mask is None:
        return actions
    values = np.asarray(actions).copy()
    width = len(mask)
    reference = np.where(mask, np.asarray(state)[..., :width], 0)
    values[..., :width] += np.expand_dims(reference, -2)
    return values


def training_action_values(actions, states, horizon, offset, mask, representation="native"):
    """Fit relative-action statistics on valid future targets, without padding."""
    actions = np.asarray(actions, dtype=np.float32)
    if mask is None and representation == "native":
        return actions[offset:]
    return np.concatenate(
        [
            encode_actions(
                actions[offset + step :, None],
                states[: len(actions) - offset - step],
                mask,
                representation,
            )[:, 0]
            for step in range(min(horizon, len(actions) - offset))
        ]
    )
