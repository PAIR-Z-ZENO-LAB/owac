"""Dataset action errors, excluding latent padding and repeated terminal actions."""

import torch


def action_sums(prediction, target, valid, statistics, execute_horizon=5):
    prediction, target = prediction[..., :7].float(), target[..., :7].float()
    error = prediction - target
    valid = valid.to(error.dtype)
    low = error.new_tensor(statistics["q01"][:7])
    scale = (error.new_tensor(statistics["q99"][:7]) - low + 1e-6) / 2
    command_error = error * scale
    predicted_grip = (prediction[..., 6] + 1) * scale[6] + low[6]
    target_grip = (target[..., 6] + 1) * scale[6] + low[6]
    square = error.square().sum(-1)
    return torch.stack(
        [
            (square * valid).double().sum(),
            (error.abs().sum(-1) * valid).double().sum(),
            (square[..., :execute_horizon] * valid[..., :execute_horizon]).double().sum(),
            (command_error[..., :3].square().sum(-1) * valid).double().sum(),
            (command_error[..., 3:6].square().sum(-1) * valid).double().sum(),
            (((predicted_grip > 0) != (target_grip > 0)) * valid).double().sum(),
            valid.double().sum(),
            valid[..., :execute_horizon].double().sum(),
        ]
    )


def action_metrics(sums):
    square, absolute, executed, translation, rotation, gripper, count, executed_count = sums
    return {
        "validation/action_mse": square / (7 * count),
        "validation/action_mae": absolute / (7 * count),
        "validation/executed_action_mse": executed / (7 * executed_count),
        "validation/translation_rmse_command": (translation / (3 * count)) ** 0.5,
        "validation/rotation_rmse_command": (rotation / (3 * count)) ** 0.5,
        "validation/gripper_sign_error": gripper / count,
        "validation/action_steps": count,
    }
