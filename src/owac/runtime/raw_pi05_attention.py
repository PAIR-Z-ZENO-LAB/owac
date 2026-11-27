"""Keep Gemma eager attention arithmetic while materializing dense GEMM batches."""

import torch
from torch.nn import functional as F


def dense_eager_attention(
    module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs
):
    # Gemma MQA broadcasts KV heads and transposes Q. Materialize these views
    # before BF16 strided-batched GEMM on Blackwell; weights and masks are unchanged.
    query = query.contiguous()
    key_states = key.repeat_interleave(module.num_key_value_groups, dim=1).contiguous()
    value_states = value.repeat_interleave(module.num_key_value_groups, dim=1).contiguous()
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
    return attn_output, attn_weights
