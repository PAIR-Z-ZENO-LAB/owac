"""Task-specific action FFNs, initialized by copying each existing shared FFN."""

import torch
from torch import nn


class TaskLinear(nn.Module):
    """Batched linear map with a task axis and no learned routing mechanism."""

    def __init__(self, source, tasks):
        super().__init__()
        self.weight = nn.Parameter(source.weight.detach().unsqueeze(0).repeat(tasks, 1, 1))
        self.bias = nn.Parameter(source.bias.detach().unsqueeze(0).repeat(tasks, 1))

    def forward(self, tokens, task_ids):
        return torch.baddbmm(
            self.bias[task_ids, None], tokens, self.weight[task_ids].transpose(1, 2)
        )


class TaskMLP(nn.Sequential):
    """Keep the legacy numeric state keys (0/2) and add only a task dimension."""

    def __init__(self, source, tasks):
        super().__init__(
            TaskLinear(source[0], tasks),
            nn.GELU(approximate=source[1].approximate),
            TaskLinear(source[2], tasks),
        )

    def forward(self, tokens, task_ids):
        return self[2](self[1](self[0](tokens, task_ids)), task_ids)


def specialize_action_mlps(generator, tasks):
    for block in generator.blocks:
        block.mlp = TaskMLP(block.mlp, tasks)


def expand_shared_mlp_weights(generator, weights):
    """Expand legacy FFNs when warm-starting; ordinary state loads stay strict."""
    expanded = {}
    for index, block in enumerate(generator.blocks):
        if not isinstance(block.mlp, TaskMLP):
            continue
        for name, target in block.mlp.state_dict().items():
            key = f"action_generator.blocks.{index}.mlp.{name}"
            source = weights[key]
            if source.ndim + 1 == target.ndim:
                weights[key] = source.unsqueeze(0).repeat(target.shape[0], *([1] * source.ndim))
                expanded[key] = {"source": list(source.shape), "target": list(target.shape)}
    return expanded
