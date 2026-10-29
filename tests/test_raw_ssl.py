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
