"""RAW-only deployment, signal branches, causal reset and semantic objective behavior."""

import copy

import pytest

torch = pytest.importorskip("torch")
from owac.baselines.raw_stem_siglip import LocalConvStem  # noqa: E402
from owac.training.raw_semantic import (  # noqa: E402
    RadiometricStem,
    RawSemanticEncoder,
    semantic_loss,
)
from tests.test_raw_jepa import Layer, encoder  # noqa: E402


def model():
    base = encoder()
    base.vision.embeddings.patch_embedding = LocalConvStem(3, 16, 2, conv_channels=4)
    base.vision.encoder.layers.append(Layer())
    result = RawSemanticEncoder(base, output_width=12, depths=(1, 2))
    result.checkpoint_blocks = False
    return result


def test_linear_initialization_preserved_and_log_branch_learns_dark_signal():
    original = LocalConvStem(3, 16, 2, conv_channels=4)
    stem = RadiometricStem(copy.deepcopy(original))
    raw = torch.rand(2, 3, 8, 8) * 0.01
    torch.testing.assert_close(stem(raw), original(raw), rtol=0, atol=0)
    stem(raw).square().mean().backward()
    assert stem.log_conv.weight.grad.abs().sum() > 0


def test_complete_encoder_gradients_and_window_reset():
    m = model().eval()
    raw = torch.rand(2, 6, 3, 8, 8)
    output = m(raw)
    assert output.shape == (2, 16, 12)
    m(torch.rand_like(raw))
    torch.testing.assert_close(m(raw), output, rtol=0, atol=0)
    torch.testing.assert_close(m(raw[:1]), output[:1])
    loss, _ = semantic_loss(output, torch.randn_like(output))
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, name
    assert m.content_gate.weight.grad.abs().sum() > 0
    assert m.vision.encoder.layers[-1].layer.linear1.weight.grad.abs().sum() > 0


def test_semantic_loss_detects_wrong_content_and_has_no_teacher_gradient():
    target = torch.randn(2, 16, 12, requires_grad=True)
    good, _ = semantic_loss(target.clone(), target)
    bad, _ = semantic_loss(target.roll(1, 1), target)
    assert bad > good
    student = torch.randn_like(target, requires_grad=True)
    loss, _ = semantic_loss(student, target)
    loss.backward()
    assert target.grad is None and student.grad is not None


def test_raw_only_artifact_state_round_trip():
    m = model().eval()
    restored = model().eval()
    restored.load_state_dict(m.state_dict())
    raw = torch.rand(2, 6, 3, 8, 8)
    torch.testing.assert_close(restored(raw), m(raw), rtol=0, atol=0)
    assert not any("teacher" in k or "dino" in k for k in m.state_dict())


def test_larger_microbatch_preserves_two_window_augmentation_and_rng():
    import random

    from owac.training.raw_semantic_train import PairedWindows

    ds = object.__new__(PairedWindows)
    ds.augmentation_group = 2
    raw = torch.linspace(0, 1, 256 * 256).reshape(1, 1, 1, 256, 256).expand(2, 6, 3, -1, -1)
    ds.native_batch = lambda keys, device: torch.cat([raw + key * 0.001 for key, _ in keys]).to(
        device
    )
    ds.rgb = lambda key, end: raw[:, 0] + key * 0.001
    keys = [(i, 5) for i in range(8)]
    random.seed(123)
    small = [ds.paired(keys[i : i + 2], "cpu", augment=True) for i in range(0, 8, 2)]
    state = random.getstate()
    random.seed(123)
    large = ds.paired(keys, "cpu", augment=True)
    assert random.getstate() == state
    for i in (0, 1):
        torch.testing.assert_close(large[i], torch.cat([x[i] for x in small]), rtol=0, atol=0)
