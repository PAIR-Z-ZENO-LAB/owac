import torch
from torch import nn

from owac.training.j5_lora import LoRALinear


def test_lora_starts_at_base_and_preserves_input_gradients():
    torch.manual_seed(17)
    base = nn.Linear(6, 4)
    layer = LoRALinear(base, rank=2, alpha=2)
    x = torch.randn(3, 6, requires_grad=True)
    torch.testing.assert_close(layer(x), base(x))
    layer(x).square().mean().backward()
    assert base.weight.grad is None
    assert x.grad.norm() > 0
    assert layer.b.weight.grad.norm() > 0


def test_lora_updates_adapter_without_changing_base():
    torch.manual_seed(19)
    layer = LoRALinear(nn.Linear(6, 4), rank=2, alpha=2)
    before = layer.base.weight.detach().clone()
    optimizer = torch.optim.AdamW([p for p in layer.parameters() if p.requires_grad])
    x = torch.randn(3, 6)
    for _ in range(2):
        optimizer.zero_grad()
        layer(x).square().mean().backward()
        optimizer.step()
    torch.testing.assert_close(layer.base.weight, before)
    assert layer.a.weight.grad.norm() > 0
    assert not torch.equal(layer(x), layer.base(x))
