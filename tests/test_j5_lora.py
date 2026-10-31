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


def test_bridge_freezes_encoder_and_policy_but_keeps_bridge_gradient():
    from owac.training.j5_lora import configure_lora

    model = nn.Module()
    model.raw_encoder = nn.Linear(6, 6)
    model.raw_projector = nn.Linear(6, 4)
    model.action_out_proj = nn.Linear(4, 2)
    model.paligemma_with_expert = nn.Module()
    model.paligemma_with_expert.joint_layers = nn.ModuleList()
    configure_lora(model, {"adaptation": {"dense_modules": ["raw_projector"], "ranks": [0, 0]}})
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.raw_projector.parameters(), lr=1e-3)
    x = torch.randn(3, 6)
    features = model.raw_encoder(x).float()
    pred = model.action_out_proj(model.raw_projector(features).to(torch.bfloat16))
    pred.float().square().mean().backward()
    optimizer.step()
    for name, p in model.named_parameters():
        if name.startswith("raw_projector."):
            assert p.grad is not None
            assert not torch.equal(p, before[name])
        else:
            assert p.grad is None
            assert torch.equal(p, before[name])


def test_selective_expert_lora_leaves_language_and_query_key_frozen():
    from owac.training.j5_lora import configure_lora

    model = nn.Module()
    model.raw_encoder = nn.Linear(4, 4)
    model.raw_projector = nn.Linear(4, 4)
    model.paligemma_with_expert = nn.Module()
    block = nn.Module()
    block.streams = nn.ModuleList()
    for _ in range(2):
        layer = nn.Module()
        layer.self_attn = nn.ModuleDict(
            {n: nn.Linear(4, 4) for n in ["q_proj", "k_proj", "v_proj", "o_proj"]}
        )
        layer.mlp = nn.ModuleDict(
            {n: nn.Linear(4, 4) for n in ["gate_proj", "up_proj", "down_proj"]}
        )
        block.streams.append(layer)
    model.paligemma_with_expert.joint_layers = nn.ModuleList([block])
    configure_lora(
        model,
        {
            "adaptation": {
                "dense_modules": ["raw_projector"],
                "ranks": [0, 2],
                "alpha_over_rank": 1,
                "targets_list": ["v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            }
        },
    )
    left, right = block.streams
    assert not any(p.requires_grad for p in left.parameters())
    assert not any(p.requires_grad for p in right.self_attn.q_proj.parameters())
    assert not any(p.requires_grad for p in right.self_attn.k_proj.parameters())
    assert isinstance(right.self_attn.v_proj, LoRALinear)
    assert right.self_attn.v_proj.b.weight.requires_grad
    assert not right.self_attn.v_proj.base.weight.requires_grad
    assert model.freeze_raw_encoder
    assert not model.raw_encoder.training
