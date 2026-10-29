"""Behavioral checks for the RAW signal boundary and distributed resume sampling."""

import torch

from owac.training.raw_ssl import RadiometricResponse, sinkhorn
from owac.training.raw_ssl_data import ResumeSampler


def test_response_preserves_hdr_and_dark_signal_and_learns_pivot():
    response = RadiometricResponse()
    raw = torch.tensor([0.0, 1 / 255, 0.1, 1.0, 4.0]).reshape(1, 1, 1, 5).expand(1, 3, 1, 5)
    out = response(raw)
    torch.testing.assert_close(out[:, :3], raw)
    assert torch.all(out[:, 3:6, :, 1:] > out[:, 3:6, :, :-1])
    torch.testing.assert_close(out[:, 6:], torch.zeros_like(raw))
    out[:, 3:6].sum().backward()
    assert response.log_pivot.grad.abs().sum() > 0


def test_resume_sampler_keeps_exact_global_sample_sequence():
    original = list(ResumeSampler(0, 5, 8, 1, 2))
    resumed = list(ResumeSampler(2, 5, 8, 1, 2))
    assert original[8:] == resumed
    other = set(ResumeSampler(0, 5, 8, 0, 2))
    assert not other.intersection(original)
    assert other.union(original) == set(range(40))


def test_sinkhorn_assignments_are_probabilities_and_ignore_common_offset():
    logits = torch.randn(128, 64)
    a = sinkhorn(logits, 0.07)
    b = sinkhorn(logits + 10, 0.07)
    torch.testing.assert_close(a.sum(-1), torch.ones(128))
    torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-4)
    assert not a.requires_grad


def test_frame_reader_matches_original_values_and_view_generation(tmp_path):
    import random
    from types import SimpleNamespace

    import numpy as np

    from owac.training.raw_ssl_data import FrameReader, RawViews, crop

    raw = np.random.default_rng(42).integers(0, 256, (5, 32, 48, 3), dtype=np.uint8)
    path = tmp_path / "raw.npz"
    np.savez(path, agentview_raw_uint8=raw)
    reader = FrameReader(path, "agentview_raw_uint8")
    for index in (4, 0, 2):
        np.testing.assert_array_equal(reader.frame(index), raw[index])
    reader.close()
    source = SimpleNamespace(path=path, domain="libero", lighting="Normal")
    dataset = RawViews([source], {})
    actual = dataset.frame(source, 2, "agentview_raw_uint8")
    expected = torch.from_numpy(raw[2].copy()).permute(2, 0, 1).float() / 255
    torch.testing.assert_close(
        crop(actual, 24, (0.32, 1), random.Random(9)),
        crop(expected, 24, (0.32, 1), random.Random(9)),
        rtol=0,
        atol=0,
    )
    dataset.close()
    floating = tmp_path / "float.npz"
    values = raw.astype(np.float32) / 32
    np.savez(floating, camera=values)
    reader = FrameReader(floating, "camera")
    np.testing.assert_array_equal(reader.frame(3), values[3])
    reader.close()
