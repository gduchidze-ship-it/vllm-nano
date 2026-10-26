"""Our own copy of Qwen2's attention.

Why copy it: step 3 replaces HF's contiguous `DynamicCache` with blocks, and you
cannot do that from the outside -- you have to own the code that reads the cache.

This file is deliberately identical to transformers' `Qwen2Attention` +
`eager_attention_forward`, written out instead of hidden behind a fused kernel:

    q, k, v      <- projections, then (batch, seq, heads, dim) -> (batch, heads, seq, dim)
    rotary       <- rotate q and k by position
    cache        <- append this step's k/v, get back everything so far
    scores       <- q @ k^T / sqrt(head_dim), plus the causal mask
    weights      <- softmax in fp32 (numerically safer), then back to the model dtype
    output       <- weights @ v -> (batch, seq, heads * dim) -> o_proj

GQA: `num_key_value_groups` query heads share each KV head, which is why K/V are
smaller than Q and must be repeated before the matmul.

Nothing here is paged yet. That is the point: prove we can reproduce the model's
numbers with readable code, *then* change where K/V live.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers.cache_utils import Cache


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(batch, kv_heads, seq, dim) -> (batch, kv_heads * n_rep, seq, dim).

    Each KV head is duplicated `n_rep` times so it can be multiplied against the
    query heads that share it. A real kernel indexes heads instead of copying.
    """
    if n_rep == 1:
        return hidden_states
    batch, kv_heads, seq_len, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(batch, kv_heads, n_rep, seq_len, head_dim)
    return expanded.reshape(batch, kv_heads * n_rep, seq_len, head_dim)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the head dimension: [a, b] -> [-b, a].

    This is half of RoPE: pairing it with cos/sin turns a position into a rotation.
    """
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rotary(
    query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to q and k. cos/sin arrive as (batch, seq, head_dim)."""
    cos = cos.unsqueeze(1)  # (batch, 1, seq, head_dim), broadcast over heads
    sin = sin.unsqueeze(1)
    return (query * cos) + (rotate_half(query) * sin), (key * cos) + (rotate_half(key) * sin)


def eager_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    scaling: float,
    num_kv_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Plain attention: softmax(q k^T * scale + mask) v.

    `attention_mask` is additive (0 where allowed, a large negative number where not),
    which is why adding it is enough -- no boolean gymnastics at this level.
    """
    key = repeat_kv(key, num_kv_groups)
    value = repeat_kv(value, num_kv_groups)

    scores = torch.matmul(query, key.transpose(2, 3)) * scaling  # (batch, heads, q, k)
    if attention_mask is not None:
        scores = scores + attention_mask

    # Softmax in fp32: with long contexts, exponentiating in fp16/bf16 loses accuracy.
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)

    output = torch.matmul(weights, value)  # (batch, heads, q, dim)
    return output.transpose(1, 2).contiguous(), weights


class Qwen2Attention(nn.Module):
    """Drop-in replacement for transformers' `Qwen2Attention`.

    Same parameters, same forward contract, so `swap_attention()` can copy the
    trained weights into it and nothing else in the model has to change.
    """

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // num_kv_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(config.hidden_size, num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: object,  # position_ids, use_cache, ... passed down by the decoder layer
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]  # (batch, seq_len)
        head_shape = (*input_shape, -1, self.head_dim)

        query = self.q_proj(hidden_states).view(head_shape).transpose(1, 2)
        key = self.k_proj(hidden_states).view(head_shape).transpose(1, 2)
        value = self.v_proj(hidden_states).view(head_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query, key = apply_rotary(query, key, cos, sin)

        # The one line that will change in the paged version: this is where the past
        # comes back. HF appends here and returns the whole contiguous history.
        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)

        attn_output, attn_weights = eager_attention(
            query,
            key,
            value,
            attention_mask,
            scaling=self.scaling,
            num_kv_groups=self.num_key_value_groups,
        )

        attn_output = self.o_proj(attn_output.reshape(*input_shape, -1).contiguous())
        return attn_output, attn_weights


def swap_attention(model) -> None:
    """Replace every attention module in `model` with ours, keeping the weights.

    In-place, because a decoder layer holds its own reference to `self_attn`.
    Use it only on a model you are happy to mutate (or on a copy of one).

    Also flips `_attn_implementation` to "eager". That flag is read by the *model*,
    not by attention: with "sdpa" the model passes `attention_mask=None` and expects
    the kernel to be causal on its own. Our module is an eager implementation, so it
    needs the additive mask the model only builds for eager. Leave the flag alone and
    attention silently becomes bidirectional -- plausible text, wrong output.
    """
    model.config._attn_implementation = "eager"

    layers = getattr(model, "model", model).layers
    for layer in layers:
        original = layer.self_attn
        replacement = Qwen2Attention(original.config, original.layer_idx)
        replacement.load_state_dict(original.state_dict())
        replacement.to(device=original.q_proj.weight.device, dtype=original.q_proj.weight.dtype)
        replacement.train(original.training)
        layer.self_attn = replacement
