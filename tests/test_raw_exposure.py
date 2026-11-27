"""Exposure geometry, noise units, and the actual backbone-gradient boundary."""

import torch

from owac.training.raw_exposure import dense_consistency, lower_exposure


def test_noise_accounts_for_variance_already_in_input():
    raw = torch.linspace(0.01, 0.9, 120).reshape(3, 5, 8)
    noise = dict(shot=4e-4, read_var=1e-5, white_level=1, levels=0)
    output = lower_exposure(raw, 2, noise, torch.Generator().manual_seed(3))
    a = 0.25
    variance = a * (1 - a) * noise["shot"] * raw + (1 - a * a) * noise["read_var"]
    expected = a * raw + variance.sqrt() * torch.randn(
        raw.shape, generator=torch.Generator().manual_seed(3)
    )
    torch.testing.assert_close(output, expected.clamp(0, 1))
    torch.testing.assert_close(lower_exposure(raw, 0, noise, torch.Generator()), raw)


def test_consistency_reaches_visible_tokens_but_not_target_or_masked_tokens():
    student = torch.randn(2, 8, 6, requires_grad=True)
    target = torch.randn(2, 8, 6, requires_grad=True)
    masks = torch.zeros(2, 8, dtype=torch.bool)
    masks[:, :3] = True
    dense_consistency(student, target, masks).backward()
    assert target.grad is None
    assert student.grad[masks].count_nonzero() == 0
    assert student.grad[~masks].abs().sum() > 0


def test_position_retrieval_uses_other_keys_in_the_same_image_and_masks_query_gradients():
    keys = torch.stack((torch.eye(4), -torch.eye(4))).requires_grad_()
    queries = keys.detach().clone().requires_grad_()
    mask = torch.tensor([[False, True, False, True], [True, False, True, False]])
    aligned = dense_consistency(queries, keys, mask, retrieval_temperature=0.07)
    expected = 0.07 * torch.log1p(3 * torch.exp(torch.tensor(-1 / 0.07)))
    torch.testing.assert_close(aligned.detach(), expected, atol=1e-7, rtol=0)
    swapped = dense_consistency(queries, keys.roll(1, dims=1), mask, retrieval_temperature=0.07)
    assert swapped > aligned
    swapped.backward()
    assert keys.grad is None
    assert queries.grad[mask].count_nonzero() == 0
    assert queries.grad[~mask].abs().sum() > 0


def test_exposure_views_share_geometry_and_resume_randomness(tmp_path):
    import json
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np

    from owac.training.raw_ssl_data import RawViews

    cfg = json.loads(Path("configs/experiments/j4_raw_exposure.json").read_text())
    cfg["exposure"]["noise"]["libero"].update(shot=0, read_var=0, levels=0)
    cfg["exposure"].update(start_stops=2, max_stops=2)
    raw = np.random.default_rng(7).integers(1, 250, (2, 64, 64, 3), dtype=np.uint8)
    path = tmp_path / "raw.npz"
    np.savez(path, agentview_raw_uint8=raw, wrist_raw_uint8=raw)
    source = SimpleNamespace(path=path, domain="libero", lighting="Normal", num_frames=2)
    dataset = RawViews([source], cfg)
    clean, local, dark = dataset[17]
    again = dataset[17]
    for left, right in zip((clean, local, dark), again, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    # No geometry changes: every pixel in each paired crop shares one gain.
    for a, b in zip(clean, dark, strict=True):
        gain = b.mean() / a.mean()
        torch.testing.assert_close(b, a * gain)
    dataset.close()
