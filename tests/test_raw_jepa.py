"""Behavior checks for masking, target isolation, causal reset and resumption."""

import random
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from owac.training.raw_jepa import (  # noqa: E402
    RawJEPA,
    RawVisualEncoder,
    SpatialPredictor,
    block_mask,
    hide_pixels,
    prediction_loss,
    update_target,
)
from owac.training.raw_jepa_train import epoch_batches  # noqa: E402


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(16, 4, 32, dropout=0, batch_first=True)

    def forward(self, hidden, attention_mask=None):
        return (self.layer(hidden),)


def encoder():
    vision = nn.Module()
    vision.config = SimpleNamespace(hidden_size=16, patch_size=2)
    vision.embeddings = nn.Module()
    vision.embeddings.patch_embedding = nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1), nn.GELU(), nn.Conv2d(4, 16, 2, stride=2)
    )
    vision.embeddings.position_embedding = nn.Embedding(16, 16)
    vision.encoder = nn.Module()
    vision.encoder.layers = nn.ModuleList([Layer()])
    vision.post_layernorm = nn.LayerNorm(16)
    return RawVisualEncoder(SimpleNamespace(vision_model=vision), checkpoint_blocks=False)


def test_mask_hides_pixels_before_overlapping_stem_in_every_frame():
    model = encoder().eval()
    raw = torch.randn(2, 6, 3, 8, 8)
    mask = block_mask(4, random.Random(101), blocks=2)
    hidden = hide_pixels(torch.ones_like(raw), mask, 2) == 0
    changed = raw.clone()
    changed[hidden] += 100
    torch.testing.assert_close(model(raw, mask), model(changed, mask), rtol=0, atol=0)
    assert not torch.equal(model(raw), model(changed))


def test_no_cross_window_state_and_no_future_frames():
    model = encoder().eval()
    raw = torch.randn(2, 6, 3, 8, 8)
    first = model(raw)
    model(torch.randn_like(raw))
    torch.testing.assert_close(first, model(raw), rtol=0, atol=0)
    torch.testing.assert_close(model(raw[:1]), first[:1])


def test_predictor_gradient_and_ema_target_isolation():
    import copy

    model = RawJEPA(encoder(), SpatialPredictor(16, width=16, depth=1, heads=4, grid=4))
    target = copy.deepcopy(model.encoder).requires_grad_(False).eval()
    raw = torch.randn(2, 6, 3, 8, 8)
    mask = block_mask(4, random.Random(4), blocks=2)
    loss, targets = prediction_loss(model(raw, mask), target(raw), mask)
    loss.backward()
    assert not targets.requires_grad
    assert all(p.grad is None for p in target.parameters())
    assert model.encoder.gate_logit.grad.abs().sum() > 0
    assert model.encoder.vision.embeddings.patch_embedding[0].weight.grad.abs().sum() > 0
    assert model.predictor.output.weight.grad.abs().sum() > 0
    before = next(target.parameters()).clone()
    with torch.no_grad():
        next(model.encoder.parameters()).add_(1)
    update_target(target, model.encoder, 0.5)
    torch.testing.assert_close(next(target.parameters()), before + 0.5)


def test_epoch_batches_have_deterministic_resume_cursor_and_full_coverage():
    keys = [(str(s), e) for s in range(5) for e in range(13)]
    batches = epoch_batches(keys, 16, 101, 3)
    assert all(len(b) == 16 for b in batches)
    assert set(keys) == {key for batch in batches for key in batch}
    assert batches[2:] == epoch_batches(keys, 16, 101, 3)[2:]
    assert batches != epoch_batches(keys, 16, 101, 4)


def test_predictor_supports_bfloat16_autocast():
    predictor = SpatialPredictor(16, width=16, depth=1, heads=4, grid=4)
    context = torch.randn(2, 8, 16)
    mask = torch.arange(16) >= 8
    with torch.autocast("cpu", dtype=torch.bfloat16):
        prediction = predictor(context, mask)
    prediction.float().square().mean().backward()
    assert predictor.input.weight.grad is not None
