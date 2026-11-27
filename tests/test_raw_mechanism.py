import torch
from torch import nn

from owac.training.raw_mechanism import encoder_class, plain_patch_state


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from types import SimpleNamespace

        self.response = nn.Module()
        self.response.log_pivot = nn.Parameter(torch.ones(1, 3, 1, 1))
        self.backbone = nn.Module()
        self.backbone.config = SimpleNamespace(num_channels=9)
        self.backbone.embeddings = nn.Module()
        self.backbone.embeddings.patch_embeddings = nn.Conv2d(9, 4, 2, stride=2)


def test_frozen_visual_readout_survives_policy_train_and_eval_transitions():
    for encoding in ("owac", "plain_patch"):
        vision = encoder_class(Encoder)(
            {"input_encoding": encoding, "freeze_for_action_readout": True}
        )
        head = nn.Linear(4, 1)
        policy = nn.ModuleDict({"vision": vision, "head": head})
        policy.eval().train()
        assert head.training and all(param.requires_grad for param in head.parameters())
        assert all(not sub.training for sub in vision.modules())
        assert all(not param.requires_grad for param in vision.parameters())
        x = torch.ones(1, vision.backbone.config.num_channels, 2, 2)
        features = vision.backbone.embeddings.patch_embeddings(x).flatten(1)
        head(features).sum().backward()
        assert head.weight.grad is not None
        assert all(param.grad is None for param in vision.parameters())


def test_plain_patch_is_linear_raw_with_valid_gradient_and_strict_reload():
    original = Encoder({})
    plain = encoder_class(Encoder)({"input_encoding": "plain_patch"})
    plain.load_state_dict(plain_patch_state(original.state_dict()), strict=True)
    raw = torch.randn(2, 3, 8, 8, requires_grad=True)
    actual = plain.backbone.embeddings.patch_embeddings(plain.response(raw))
    expected = original.backbone.embeddings.patch_embeddings(
        torch.cat((raw, torch.zeros_like(raw), torch.zeros_like(raw)), 1)
    )
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    assert raw.grad.abs().sum() > 0
    assert plain.backbone.embeddings.patch_embeddings.weight.grad.abs().sum() > 0
    assert not any("log_pivot" in key for key in plain.state_dict())


def test_owac_control_leaves_initialization_and_rng_unchanged():
    torch.manual_seed(19)
    expected = Encoder({})
    expected_rng = torch.get_rng_state()
    torch.manual_seed(19)
    actual = encoder_class(Encoder)({})
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[key], value, rtol=0, atol=0)


def test_random_patch_does_not_truncate_and_preserves_shared_rng():
    torch.manual_seed(73)
    full = encoder_class(Encoder)({})
    after_full = torch.randn(8)
    torch.manual_seed(73)
    plain = encoder_class(Encoder)(
        {"input_encoding": "plain_patch", "patch_initialization": "random"}
    )
    after_plain = torch.randn(8)
    assert torch.equal(after_full, after_plain)
    assert not torch.equal(
        full.backbone.embeddings.patch_embeddings.weight[:, :3],
        plain.backbone.embeddings.patch_embeddings.weight,
    )
