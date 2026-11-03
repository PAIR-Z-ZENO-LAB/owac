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


def test_residual_bridge_preserves_old_mapping_and_learns_correction():
    from owac.training.j5_lora import ResidualProjector

    torch.manual_seed(31)
    base = nn.Linear(6, 4)
    bridge = ResidualProjector(base, hidden=8)
    x = torch.randn(3, 6)
    torch.testing.assert_close(bridge(x), base(x))
    bridge(x).square().mean().backward()
    assert bridge.residual[-1].weight.grad.norm() > 0
    assert bridge.base.weight.grad.norm() > 0


def test_expanding_lora_preserves_existing_adapters():
    from owac.training.j5_lora import configure_lora, parameter_groups

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
    cfg = {
        "adaptation": {
            "dense_modules": ["raw_projector"],
            "ranks": [0, 2],
            "alpha_over_rank": 1,
            "targets_list": ["v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        }
    }
    configure_lora(model, cfg)
    old = block.streams[1].self_attn.v_proj
    with torch.no_grad():
        old.b.weight.fill_(0.125)
    cfg["adaptation"].update(ranks=[2, 2], residual_projector=True)
    del cfg["adaptation"]["targets_list"]
    configure_lora(model, cfg)
    assert block.streams[1].self_attn.v_proj is old
    assert old.a.weight.requires_grad and old.b.weight.requires_grad
    assert torch.all(old.b.weight == 0.125)
    assert torch.count_nonzero(block.streams[1].self_attn.q_proj.b.weight) == 0
    training = {
        "learning_rate": 3e-5,
        "group_learning_rates": {
            "bridge": 3e-5,
            "language": 5e-6,
            "expert_content": 5e-6,
            "expert_qk": 1e-6,
        },
    }
    groups = parameter_groups(model, training)
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in model.parameters() if p.requires_grad}


def test_unfrozen_encoder_receives_action_gradient_in_its_own_lr_group():
    from owac.training.j5_lora import configure_lora, parameter_groups

    model = nn.Module()
    model.raw_encoder = nn.Linear(6, 6)
    model.raw_projector = nn.Linear(6, 4)
    model.action_out_proj = nn.Linear(4, 2)
    model.paligemma_with_expert = nn.Module()
    model.paligemma_with_expert.joint_layers = nn.ModuleList()
    configure_lora(
        model, {"adaptation": {"dense_modules": ["raw_encoder", "raw_projector"], "ranks": [0, 0]}}
    )
    groups = parameter_groups(
        model, {"learning_rate": 3e-5, "group_learning_rates": {"vision": 1e-6, "bridge": 3e-5}}
    )
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = torch.optim.AdamW(groups, lr=1e-3)
    pred = model.action_out_proj(
        model.raw_projector(model.raw_encoder(torch.randn(3, 6))).to(torch.bfloat16)
    )
    pred.float().square().mean().backward()
    optimizer.step()
    assert not model.freeze_raw_encoder
    assert model.raw_encoder.training
    assert model.raw_encoder.weight.grad.norm() > 0
    assert not torch.equal(model.raw_encoder.weight, before["raw_encoder.weight"])
    assert torch.equal(model.action_out_proj.weight, before["action_out_proj.weight"])
    assert {id(p) for p in groups[0]["params"]} == {id(p) for p in model.raw_encoder.parameters()}
    assert groups[0]["lr_scale"] == 1e-6 / 3e-5
