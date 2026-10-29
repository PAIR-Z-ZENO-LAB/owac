import io
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from owac.backends.rawvla_features_n02 import build_stub_bridge
from owac.training.raw_action_adaptation import (
    BudgetWriter,
    native_flow_loss,
    normalize_window,
    save_budgeted,
)


def test_current_action_alignment_terminal_mask_and_padding():
    actions = np.arange(12 * 7, dtype=np.float32).reshape(12, 7) / 100
    states = np.zeros((12, 8), np.float32)
    stats = {k: {"q01": [-1] * w, "q99": [1] * w} for k, w in (("actions", 7), ("state", 8))}
    state, padded, valid = normalize_window(actions, states, 8, stats)
    np.testing.assert_allclose(padded[:4, :7], actions[8:], atol=1e-7)
    assert valid.tolist() == [True] * 4 + [False] * 6
    assert not padded[:, 7:].any()
    assert state.shape == (8,)


def test_native_flow_gradient_frozen_backend_and_tail_mask(monkeypatch):
    from owac.backends import rawvla_language

    bridge = build_stub_bridge(
        width=8,
        token_dim=8,
        tokens_per_camera=256,
        frame_pixels=16,
        cameras=("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
        with_frame=False,
    )["bridge"]

    def language(policy, prompts, proprio, device):
        return (
            torch.ones(len(prompts), 2, dtype=torch.long),
            torch.ones(len(prompts), 2, dtype=torch.bool),
            torch.zeros(len(prompts), 32),
        )

    monkeypatch.setattr(rawvla_language, "prepare_language_state", language)
    visual = torch.randn(1, 512, 8, requires_grad=True)
    target = SimpleNamespace(
        state=np.zeros(8),
        actions=np.zeros((10, 32), np.float32),
        valid=np.array([True] * 3 + [False] * 7),
        prompt="move",
    )
    noise = torch.randn(1, 10, 32)
    tau = torch.tensor([0.5])
    total, count = native_flow_loss(
        bridge, visual, torch.ones(1, 512, dtype=torch.bool), [target], noise=noise, flow_time=tau
    )
    assert count.item() == 3
    (total / count).backward()
    assert visual.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in bridge.base.parameters())
    assert bridge.base.grad_enabled_during_train == [True]
    reference = bridge.forward_from_visual_tokens(
        SimpleNamespace(
            values=visual.detach(),
            mask=torch.ones(1, 512, dtype=torch.bool),
            camera_ids=("base_0_rgb", "left_wrist_0_rgb"),
            positions=None,
        ),
        torch.ones(1, 2, dtype=torch.long),
        torch.ones(1, 2, dtype=torch.bool),
        torch.zeros(1, 32),
        torch.zeros(1, 10, 32),
        noise=noise,
        time=tau,
        action_mask=torch.as_tensor(target.valid)[None],
    )
    torch.testing.assert_close(total.detach() / count, reference["action_loss"])


def test_budget_atomic_checkpoint_roundtrip_and_failed_write_keeps_latest(tmp_path):
    path = tmp_path / "latest.pt"
    save_budgeted(path, {"model": torch.ones(3), "global_step": 1}, tmp_path, 0, 100000)
    assert torch.load(path, weights_only=True)["global_step"] == 1
    original = path.read_bytes()
    with pytest.raises((OSError, RuntimeError)):
        save_budgeted(path, {"model": torch.ones(3000)}, tmp_path, 0, len(original) + 100)
    assert path.read_bytes() == original
    stream = io.BytesIO()
    writer = BudgetWriter(stream, 3)
    writer.write(b"abc")
    with pytest.raises(OSError):
        writer.write(b"d")
