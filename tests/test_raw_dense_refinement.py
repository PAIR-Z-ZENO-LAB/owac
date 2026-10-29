"""Geometry, fixed-anchor isolation and residual initialization behavior."""

import copy
import random

import pytest

torch = pytest.importorskip("torch")
from owac.training.raw_dense_refinement import (  # noqa: E402
    crop_correspondence,
    dense_objective,
    fixed_spatial_anchor,
    overlapping_crops,
)
from tests.test_raw_jepa import Layer, encoder  # noqa: E402


def refined_encoder():
    model = encoder()
    model.vision.encoder.layers.append(Layer())
    model.refinement_layer = 1
    model.refinement_scale = torch.nn.Parameter(torch.zeros(()))
    return model


def test_zero_residual_preserves_measured_layer_and_gate_gets_gradient():
    model = refined_encoder().eval()
    raw = torch.randn(2, 6, 3, 8, 8)
    torch.testing.assert_close(model(raw), model(raw, stop_at=1), rtol=0, atol=0)
    model(raw).square().mean().backward()
    assert model.refinement_scale.grad.abs() > 0
    model.zero_grad()
    with torch.no_grad():
        model.refinement_scale.fill_(0.1)
    model(raw).square().mean().backward()
    assert next(model.vision.encoder.layers[-1].parameters()).grad.abs().sum() > 0


def test_anchor_is_fixed_independent_and_has_no_deep_blocks():
    model = refined_encoder().eval()
    anchor = fixed_spatial_anchor(model, 1)
    raw = torch.randn(2, 6, 3, 8, 8)
    expected = model(raw, stop_at=1).detach()
    torch.testing.assert_close(anchor(raw), expected)
    with torch.no_grad():
        next(model.parameters()).add_(1)
    torch.testing.assert_close(anchor(raw), expected)
    assert len(anchor.vision.encoder.layers) == 1
    assert all(not p.requires_grad for p in anchor.parameters())


def test_crop_mapping_tracks_content_in_both_axes_and_all_frames():
    native = torch.arange(256 * 256).reshape(1, 1, 1, 256, 256).expand(2, 6, 3, -1, -1)
    views, ia, ib = overlapping_crops(native, random.Random(10))
    patches = [v.unfold(-2, 14, 14).unfold(-2, 14, 14).flatten(-4, -3) for v in views]
    torch.testing.assert_close(patches[0][..., ia, :, :], patches[1][..., ib, :, :], rtol=0, atol=0)
    for origins in [((0, 0), (28, 0)), ((0, 0), (0, 28)), ((28, 28), (0, 0))]:
        a, b = crop_correspondence(*origins)
        assert len(a) > 0 and len(a) == len(b)


def test_content_matching_beats_stationary_shortcut_and_detaches_anchor():
    # Physical tokens match after the crop shift; a fixed coordinate code does not.
    ia, ib = crop_correspondence((0, 0), (0, 14), size=84, patch=14)
    a = torch.eye(36)[None].requires_grad_()
    b = torch.roll(torch.eye(36), -1, 0)[None].requires_grad_()
    aa = copy.deepcopy(a)
    ab = copy.deepcopy(b)
    ce, gram, matches = dense_objective(a, b, aa, ab, ia, ib)
    stationary, _, _ = dense_objective(a, a, aa, aa, ia, ib)
    assert ce < stationary
    assert matches == 1 and gram == 0
    (ce + gram).backward()
    assert a.grad is not None and b.grad is not None
    assert aa.grad is None and ab.grad is None
