import torch

from owac.training.j5_action_metrics import action_metrics, action_sums
from owac.training.j5_monitoring import canonical


def test_mse_excludes_latent_and_terminal_padding_and_uses_command_scales():
    target = torch.zeros(1, 10, 32)
    prediction = torch.ones_like(target)
    prediction[..., 7:] = 1000
    prediction[:, 2:, :7] = 1000
    valid = torch.tensor([[True, True] + [False] * 8])
    stats = {"q01": [-1] * 7, "q99": [3] * 7}
    result = action_metrics(action_sums(prediction, target, valid, stats).tolist())
    assert result["validation/action_mse"] == 1
    assert result["validation/action_mae"] == 1
    assert result["validation/executed_action_mse"] == 1
    assert result["validation/action_steps"] == 2
    # Both normalized 0 and 1 unnormalize to positive gripper commands.
    assert result["validation/gripper_sign_error"] == 0
    torch.testing.assert_close(
        torch.tensor(result["validation/translation_rmse_command"]), torch.tensor(2.0000005)
    )


def test_global_error_is_weighted_by_real_actions_not_mean_of_clip_means():
    stats = {"q01": [-1] * 7, "q99": [1] * 7}
    a = action_sums(
        torch.ones(1, 2, 32), torch.zeros(1, 2, 32), torch.tensor([[True, False]]), stats
    )
    b = action_sums(
        torch.full((1, 2, 32), 3.0),
        torch.zeros(1, 2, 32),
        torch.ones(1, 2, dtype=torch.bool),
        stats,
    )
    result = action_metrics((a + b).tolist())
    assert result["validation/action_mse"] == 19 / 3


def test_monitoring_drops_duplicate_lr_and_does_not_relabel_mae_as_mse():
    result = canonical(
        {
            "train/lr": 0.1,
            "train/lr_bridge": 0.1,
            "train/images_per_second": 40,
            "validation/action_mae": 0.3,
        }
    )
    assert result == {
        "optimizer/lr_bridge": 0.1,
        "runtime/examples_per_second": 40,
        "validation/action_mae": 0.3,
    }
