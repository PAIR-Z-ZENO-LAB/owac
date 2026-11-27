from types import SimpleNamespace

import numpy as np
import pytest
import torch

from owac.frameworks.raw_compact.factory import make_policy
from owac.training.fruit_grasp import approach_masks, banana_image_errors
from owac.training.teleop_policy_data import (
    FruitFrames,
    command_chunks,
    fruit_exposure,
    gripper_travel,
    paired_low_light,
)
from owac.training.teleop_policy_eval import (
    command_targets,
    evaluate,
    predict_actions,
    predict_commands,
)
from tests.real_model_fixtures import example, measurement, native_sensor


def test_checkpoint_selection_weights_episodes_equally_and_ignores_padded_commands():
    targets = torch.tensor([1, 50, 3, 4, 7, 10], dtype=torch.float32)
    absolute = targets[:, None, None].expand(-1, 2, 8).clone()
    absolute[2, 1] = 1000
    valid = torch.ones(6, 2, dtype=torch.bool)
    valid[2, 1] = False
    bank = SimpleNamespace(
        cfg={"model": {"action_horizon": 2}, "evaluation": {"selection_anchors_per_episode": 2}},
        sources=[SimpleNamespace(group_id=name, suite="bowl") for name in ("first", "second")],
        starts=np.array([0, 3]),
        sizes=np.array([3, 3]),
        episode=np.repeat([0, 1], 3),
        tensor=torch.as_tensor,
        device="cpu",
        state=torch.zeros(6, 8),
        absolute=absolute,
        normalized=absolute,
        valid=valid,
        low=torch.full((8,), -1.0),
        scale=torch.full((8,), 2.0),
        relative_mask=torch.ones(8),
    )
    metrics, per = evaluate(None, bank, method="hold")
    # Episode one has three valid commands at its end anchors. Episode two has four.
    assert metrics["selection_normalized_l1"] == pytest.approx(((1 + 1 + 3) / 3 + 7) / 2)
    assert metrics["macro_normalized_l1"] == pytest.approx(
        (per["first"]["normalized_l1"] + per["second"]["normalized_l1"]) / 2
    )


def test_physical_gripper_width_maps_subzero_predictions_to_closed_contact():
    raw = np.array([[-0.2] * 7 + [-0.013], [0.4] * 7 + [0.082]], dtype=np.float32)
    commands = command_targets(raw)
    np.testing.assert_array_equal(commands[:, :7], raw[:, :7])
    np.testing.assert_array_equal(commands[:, 7], [0, raw[1, 7]])
    assert raw[0, 7] < 0


def test_live_prediction_preserves_signed_width_and_extrapolated_joint_targets():
    model = torch.nn.Linear(1, 1)
    model.cfg = {
        "observation": {
            "domain": "real_teleop",
            "raw_shape": [2, 2],
            "relative_action_mask": [True] * 7 + [False],
        },
        "statistics": dict(
            state_mean=[0] * 8, state_std=[1] * 8, action_q01=[-1] * 8, action_q99=[1] * 8
        ),
        "task_token_ids": [0],
        "task_token_mask": [True],
    }
    model.sample = lambda batch: torch.tensor([3.0] * 7 + [-0.025]).expand(
        len(batch["state"]), 12, 8
    )
    raw = np.zeros((2, 2, 2), dtype=np.uint16)
    state = np.ones(8, dtype=np.float32)
    predicted = predict_actions(model, raw, state)
    np.testing.assert_allclose(predicted[..., :7], 4.0, atol=3e-6)
    np.testing.assert_allclose(predicted[..., 7], -0.025, atol=1e-6)
    np.testing.assert_array_equal(predict_commands(model, raw, state), command_targets(predicted))


def test_matched_darkening_keeps_native_counts_and_aligns_both_views_to_the_source_light():
    raw = torch.full((1, 2, 64, 208), 144.0)
    raw[:, 1] = 104
    raw[:, :, 0, 0] = 64
    fit = {"counts": torch.ones(1), "levels": torch.tensor([68.0]), "sigma": torch.zeros(2, 4)}
    result = paired_low_light(raw, {"black_level": 64, "white_level": 1023}, fit, torch.Generator())
    assert result[0, :, 12, 12].tolist() == [68, 66]
    assert result[0, :, 0, 0].tolist() == [64, 64]
    assert raw[0, :, 12, 12].tolist() == [144, 104]


def test_stereo_exposure_preserves_bayer_coordinates_black_level_and_clean_examples():
    sensor = dict(layout="bayer", units="dn", black_level=64, white_level=1023, shot=0, read_var=0)
    cfg = {"model": {"raw_encoder": {"measurement_encoder": {"sensor": sensor}}}}
    mosaic = torch.tensor([[64, 128], [256, 512]]).expand(3, 2, 2, 2)
    result = fruit_exposure(
        mosaic, cfg, torch.tensor([0.0, 1.0, 2.0]).reshape(-1, 1, 1, 1), torch.Generator()
    )
    for i, divisor in enumerate([1, 2, 4]):
        torch.testing.assert_close(result[i], 64 + (mosaic[i] - 64) / divisor)
    assert torch.equal(result[:, 0], result[:, 1])


def test_gripper_motion_sampler_preserves_episode_balance_and_excludes_padded_steps():
    from types import SimpleNamespace

    actions = torch.zeros(3, 4, 8)
    actions[:, :, 7] = torch.tensor([0, 1, 3, 50])
    valid = torch.tensor([[True, True, True, False]] * 3)
    torch.testing.assert_close(
        gripper_travel(actions, valid)[0], torch.tensor([0.0, 1.0, 2.0, 0.0])
    )
    bank = SimpleNamespace(
        sources=[1, 2],
        starts=np.array([0, 5]),
        sizes=np.array([5, 5]),
        tensor=np.asarray,
        cfg={"training": {"grip_motion_fraction": 1}},
        motion_cdf=[np.array([0, 1, 1, 1, 1]), None],
        approach_rows=[],
        placement_rows=[],
    )
    selected = FruitFrames.sample_indices(bank, seed=41, batch=128)
    assert set(selected) == {1, 5, 6, 7, 8, 9}
    same = np.random.default_rng(41).integers(2, size=128)
    np.testing.assert_array_equal(selected >= 5, same)


def test_grasp_metrics_only_score_action_times_before_and_at_closure():
    time = np.array([2, 1, 0.099, 0.0, -0.01])
    valid = np.ones((5, 4), bool)
    near, close = approach_masks(time, valid, 10, 1.5)
    assert near.sum(1).tolist() == [0, 4, 1, 1, 0]
    assert close.sum(1).tolist() == [0, 0, 1, 1, 0]
    p = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0]])
    assert banana_image_errors(p, p, (2, 3), 16) == {"centre_px": 0.0, "axis_deg": 0.0}


def test_banana_head_starts_with_parent_actions_and_uses_only_raw_for_its_geometry():
    torch.set_num_threads(2)
    cfg, batch = example("libero")
    cfg["model"].update(kind="corp_regression", action_head="regression", state_dim=8, action_dim=8)
    cfg["observation"]["raw_shape"] = [32, 48]
    cfg["model"]["raw_encoder"]["measurement_encoder"] = measurement(
        native_sensor("bayer", pattern="BGGR", origin=[0, 0]), evidence=False
    )
    batch.update(
        raw=torch.randint(0, 1024, (2, 2, 32, 48)).float(),
        state=torch.randn(2, 8),
        actions=torch.randn(2, 3, 8),
        action_valid=torch.ones(2, 3, dtype=torch.bool),
        task_id=torch.tensor([0, 1]),
    )
    parent = make_policy(cfg).eval()
    cfg["model"].update(
        banana_grounding=True, spatial_goal_head={"width": 16, "arms": 1}, spatial_coordinates=True
    )
    child = make_policy(cfg).eval()
    missing, _ = child.load_state_dict(parent.state_dict(), strict=False)
    assert "banana_projection.weight" in missing
    torch.testing.assert_close(parent.sample(batch), child.sample(batch))
    loss, _, _, with_pose = child(batch, return_prediction=True)
    torch.testing.assert_close(with_pose, child.sample(batch))
    loss.backward()
    assert child.banana_projection.weight.grad.norm() > 0
    child.zero_grad(set_to_none=True)
    _, _, logits = child.predict(batch, return_goal=True)
    batch["banana_mass"] = torch.zeros_like(logits)
    batch["banana_mass"][:, 0] = 1
    batch["banana_visible"] = torch.ones(2)
    initial = child.sample(batch)
    aux = child(batch, return_prediction=True)[4]
    aux.backward()
    assert child.spatial_goal_head.key[-1].weight.grad.norm() > 0
    batch["banana_mass"] = batch["banana_mass"].flip(-1)
    torch.testing.assert_close(initial, child.sample(batch))


def test_timestamped_commands_mask_episode_end_and_choose_first_after_each_offset():
    ns = np.array([0, 55, 91, 160]) * 1_000_000
    act = np.arange(32, dtype=np.float32).reshape(4, 8)
    y, valid = command_chunks(np.array([4, 91]) * 1_000_000, ns, act, 3, 20)
    np.testing.assert_array_equal(y[0], act[[1, 1, 3]])
    np.testing.assert_array_equal(y[1, :2], act[[2, 3]])
    np.testing.assert_array_equal(valid, [[True, True, True], [True, True, False]])


@pytest.mark.parametrize("kind", ["corp_regression", "corp"])
def test_real_encoder_tokens_and_raw_give_same_regression_and_ignore_padded_labels(kind):
    torch.set_num_threads(2)
    cfg, batch = example("libero")
    cfg["model"].update(kind=kind, action_head="regression", state_dim=8, action_dim=8)
    cfg["observation"]["raw_shape"] = [32, 48]
    cfg["model"]["raw_encoder"]["measurement_encoder"] = measurement(
        native_sensor("bayer", pattern="BGGR", origin=[0, 0]), evidence=False
    )
    model = make_policy(cfg).eval()
    batch["raw"] = torch.randint(0, 1024, (2, 2, 32, 48)).float()
    batch["state"] = torch.randn(2, 8)
    batch["actions"] = torch.randn(2, 3, 8)
    batch["action_valid"] = torch.tensor([[True, True, False], [True, False, False]])
    batch["task_id"] = torch.tensor([0, 1])
    with torch.no_grad():
        _, tokens = model.encode_raw(batch["raw"])
        cached = {k: v for k, v in batch.items() if k != "raw"}
        cached["features"] = torch.stack(tokens.chunk(2), dim=1)
        target = model.sample(batch, torch.randn_like(batch["actions"]))
        torch.testing.assert_close(model.sample(cached, torch.zeros_like(batch["actions"])), target)
        loss = model(batch)
        batch["actions"][~batch["action_valid"]] = torch.nan
        torch.testing.assert_close(model(batch), loss)
