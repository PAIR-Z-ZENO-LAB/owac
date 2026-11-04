"""Behavioral checks for action masking, temporal alignment and modality paths."""

import copy

import numpy as np
import pytest
import torch

from owac.frameworks.raw_compact._vendor.action_generator import DiTActionGenerator
from owac.frameworks.raw_compact._vendor.representation import Representation
from owac.frameworks.raw_compact.policy import masked_flow_loss
from owac.training.compact_data import normalized_chunk, unnormalize_actions


def config():
    return {
        "hidden_dim": 48,
        "action_dim": 7,
        "action_horizon": 3,
        "action_generator": {
            "num_layers": 2,
            "num_heads": 4,
            "mlp_ratio": 2,
            "max_horizon": 8,
            "adaln_rank": 8,
            "num_sample_steps": 10,
        },
    }


def test_future_chunk_starts_at_observation_and_masks_terminal_padding():
    actions = np.arange(28, dtype=np.float32).reshape(4, 7) / 20
    statistics = {"action_q01": [-1] * 7, "action_q99": [1] * 7}
    normalized, valid = normalized_chunk(actions, 2, 4, statistics)
    restored = unnormalize_actions(normalized, statistics)
    np.testing.assert_allclose(restored[:2], actions[2:], atol=2e-7)
    assert valid.tolist() == [True, True, False, False]
    prediction = torch.randn(1, 4, 7, requires_grad=True)
    target = torch.from_numpy(normalized)[None]
    loss = masked_flow_loss(prediction, target, torch.from_numpy(valid)[None])
    loss.backward()
    assert prediction.grad[:, :2].abs().sum() > 0
    assert prediction.grad[:, 2:].count_nonzero() == 0


def test_prefix_padding_is_ignored_by_all_layers_and_action_queries():
    torch.manual_seed(2)
    model = DiTActionGenerator(config())
    tokens = torch.randn(2, 6, 48)
    language = torch.randn(2, 48)
    valid = torch.tensor([[True, True, True, False, False, True]]).expand(2, -1)
    noisy, tau = torch.randn(2, 3, 7), torch.rand(2)
    original = Representation(tokens, language, torch.float32, valid)
    modified = tokens.clone()
    modified[:, 3:5] = torch.randn_like(modified[:, 3:5]) * 100
    replacement = Representation(modified, language, torch.float32, valid)
    first = model(noisy, tau, model.prepare_prefix(original))
    second = model(noisy, tau, model.prepare_prefix(replacement))
    torch.testing.assert_close(first, second)


def test_cached_prefix_has_gradient_and_is_independent_of_action_noise():
    torch.manual_seed(3)
    model = DiTActionGenerator(config())
    prefix = torch.randn(2, 5, 48, requires_grad=True)
    language = torch.randn(2, 48, requires_grad=True)
    representation = Representation(prefix, language, torch.float32)
    cache = model.prepare_prefix(representation)
    original = [item.key.clone() for item in cache]
    output = model(torch.randn(2, 3, 7), torch.rand(2), cache)
    output.square().mean().backward()
    assert prefix.grad.abs().sum() > 0
    assert language.grad.abs().sum() > 0
    for key, item in zip(original, cache, strict=True):
        torch.testing.assert_close(key, item.key)


def test_full_policy_uses_raw_state_and_language():
    pytest.importorskip("transformers")
    from owac.frameworks.raw_compact.policy import CompactPolicy

    cfg = {
        "model": config()
        | {
            "raw_encoder": {
                "width": 384,
                "depth": 1,
                "heads": 6,
                "patch_size": 16,
                "image_size": 32,
                "drop_path": 0.0,
                "response_pivot": 0.02,
            },
            "visual_initialization": "random",
            "language": {
                "vocab_size": 24,
                "d_model": 32,
                "d_ff": 64,
                "d_kv": 8,
                "num_heads": 4,
                "num_layers": 1,
                "dropout_rate": 0.0,
                "pad_token_id": 0,
                "eos_token_id": 1,
            },
            "tokens_per_view": 2,
            "resampler_heads": 4,
            "film_rank": 8,
        },
        "task_names": ["a", "b"],
        "training": {"task_loss_weight": 0.01},
    }
    model = CompactPolicy(cfg).eval()
    batch = {
        "raw": torch.rand(2, 2, 3, 32, 32),
        "state": torch.randn(2, 9),
        "tokens": torch.tensor([[2, 3, 1, 0], [3, 2, 1, 0]]),
        "token_mask": torch.tensor([[True, True, True, False]]).expand(2, -1),
        "actions": torch.randn(2, 3, 7),
        "action_valid": torch.ones(2, 3, dtype=torch.bool),
        "task_id": torch.tensor([0, 1]),
    }
    noise = torch.randn(2, 3, 7)
    base = model.sample(batch, noise, steps=2)
    for key in ["raw", "state", "tokens"]:
        changed = copy.copy(batch)
        changed[key] = batch[key].flip(0)
        assert not torch.allclose(base, model.sample(changed, noise, steps=2))
    loss, _, _ = model(batch, noise=noise, tau=torch.tensor([0.2, 0.7]))
    loss.backward()
    for module in [
        model.raw_encoder,
        model.language_encoder,
        model.resampler,
        model.state_encoder,
        model.action_generator,
    ]:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
