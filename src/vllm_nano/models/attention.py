"""Our own copy of Qwen2's attention, plus a paged version of it.

Why copy it: step 3 replaces HF's contiguous `DynamicCache` with blocks, and you
cannot do that from the outside -- you have to own the code that reads the cache.

Everything up to the cache is shared and deliberately identical to transformers'
`Qwen2Attention`:

    q, k, v      <- projections, then (batch, seq, heads, dim) -> (batch, heads, seq, dim)
    rotary       <- rotate q and k by position

From there the cache decides which of two functions runs:

    eager_attention   the whole score matrix at once, normalized once. Needs the
                      full history as one contiguous (batch, heads, length, dim).
    paged_attention   walks the history block by block and merges each block into a
                      running softmax. Never builds the history, and never builds the
                      full score matrix either.

GQA: `num_key_value_groups` query heads share each KV head, which is why K/V are
smaller than Q and must be repeated before the matmul.

The two paths are mathematically identical and differ in the last bits, because
floating-point addition is not associative and the paged loop adds in a different
order. Token-level output must still match Hugging Face exactly.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from transformers.cache_utils import Cache

from vllm_nano.models.paged_cache import PagedCache


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


def paged_attention(
    query: torch.Tensor,
    cache: PagedCache,
    layer_idx: int,
    attention_mask: torch.Tensor | None,
    *,
    scaling: float,
    num_kv_groups: int,
) -> tuple[torch.Tensor, None]:
    """Attention that walks the cache block by block, with a running softmax.

    Same result as `eager_attention`, organized so the history never exists:

        for each block:
            scores = q @ K_block^T * scale + mask_slice
            merge it into (running max, running sum, running weighted V)

    You cannot normalize until you have seen every score, and you do not want to keep
    every score -- so keep three accumulators and rescale them whenever a bigger
    maximum shows up:

        new_max    = max(running_max, max(scores))
        correction = exp(running_max - new_max)     # 1 for the first block, <= 1 later
        running_out = running_out * correction + exp(scores - new_max) @ V
        running_sum = running_sum * correction + sum(exp(scores - new_max))

    `running_out / running_sum` is the same softmax average as computing everything at
    once. The trick is that the cost per block is fixed, so memory stays flat and the
    sequence length only changes the number of iterations.

    Precision: the accumulators are fp32 (as in flash-attention) and the result is cast
    back to the query dtype. Reordered additions mean the last bits differ from
    `eager_attention`; the math does not.

    Masks here are additive but *finite* (HF uses the dtype's most negative value, not
    -inf), so a query that can see nothing averages V uniformly -- the same answer the
    contiguous path gives -- instead of producing a NaN.
    """
    length = cache.step_length()
    if length == 0:
        raise RuntimeError("paged attention ran before any K/V was written")

    block_size = cache.block_size
    batch, heads, q_len, head_dim = query.shape

    # One accumulator set per (row, head, query position). Independent of `length`.
    running_max = torch.full(
        (batch, heads, q_len, 1), -torch.inf, dtype=torch.float32, device=query.device
    )
    running_sum = torch.zeros((batch, heads, q_len, 1), dtype=torch.float32, device=query.device)
    running_out = torch.zeros(
        (batch, heads, q_len, head_dim), dtype=torch.float32, device=query.device
    )

    for block_index in range(math.ceil(length / block_size)):
        start = block_index * block_size
        take = min(block_size, length - start)  # the last block is usually partial

        key, value = cache.block(layer_idx, block_index)  # (batch, kv_heads, bs, dim)
        key = repeat_kv(key[:, :, :take], num_kv_groups)
        value = repeat_kv(value[:, :, :take], num_kv_groups).float()

        scores = torch.matmul(query, key.transpose(2, 3)).float() * scaling
        if attention_mask is not None:
            # The mask is indexed by cache position, so block b owns columns
            # [start, start + take). This alignment is the easy thing to get wrong.
            scores = scores + attention_mask[..., start : start + take]

        block_max = scores.amax(dim=-1, keepdim=True)
        new_max = torch.maximum(running_max, block_max)
        correction = torch.exp(running_max - new_max)
        weights = torch.exp(scores - new_max)

        running_out = running_out * correction + torch.matmul(weights, value)
        running_sum = running_sum * correction + weights.sum(dim=-1, keepdim=True)
        running_max = new_max

    output = (running_out / running_sum).to(query.dtype)
    # Weights were never materialized, so there is nothing to return for them.
    return output.transpose(1, 2).contiguous(), None


class Qwen2Attention(nn.Module):
    """Drop-in replacement for transformers' `Qwen2Attention`.

    Same parameters, same forward contract, so `swap_attention()` can copy the
    trained weights into it and nothing else in the model has to change.

    Two caches are supported: HF's `DynamicCache` (attention gets the contiguous
    history) and our `PagedCache` (attention reads blocks). The math is the same;
    only the memory layout and the order of additions differ.
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
        past_key_values: Cache | PagedCache | None = None,
        **kwargs: object,  # position_ids, use_cache, ... passed down by the decoder layer
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]  # (batch, seq_len)
        head_shape = (*input_shape, -1, self.head_dim)

        query = self.q_proj(hidden_states).view(head_shape).transpose(1, 2)
        key = self.k_proj(hidden_states).view(head_shape).transpose(1, 2)
        value = self.v_proj(hidden_states).view(head_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query, key = apply_rotary(query, key, cos, sin)

        if isinstance(past_key_values, PagedCache):
            # Store this layer's K/V in blocks, then read the history from them.
            # The gather is gone: what gets stored is what attention multiplies.
            past_key_values.update(key, value, self.layer_idx)
            attn_output, attn_weights = paged_attention(
                query,
                past_key_values,
                self.layer_idx,
                attention_mask,
                scaling=self.scaling,
                num_kv_groups=self.num_key_value_groups,
            )
        else:
            # Contiguous path: the cache appends and hands the whole history back.
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
