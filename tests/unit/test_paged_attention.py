"""Paged attention: the same numbers, read out of blocks instead of one buffer.

The claim is strict -- paging changes addresses, not values -- so these tests compare
with `torch.equal` rather than `allclose`. A tolerance would happily hide an offset
that is wrong by one token, which is the exact bug paging invites.

The reference is the `DynamicCache` path through the same hand-written attention.
That path is already proven equal to Hugging Face in `test_attention.py`, so equality
here means equality all the way to the real model.
"""

import copy
import math

import pytest
import torch

from vllm_nano.core.kv_cache import BlockAllocator, BlockTable, PagedKVCache
from vllm_nano.models.attention import swap_attention
from vllm_nano.models.paged_cache import PagedCache

IDS = torch.tensor([[1, 2, 3, 4, 5]])
BLOCK = 3  # small on purpose: prefill and decode both cross block boundaries
LAYERS = 2
KV_HEADS = 2
HEAD_DIM = 8
DECODE_STEPS = 3


def make_step(n_tokens: int, seed: int = 0) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """One step's K/V, with a different generator per layer so swaps are visible."""
    keys, values = [], []
    for layer in range(LAYERS):
        gen = torch.Generator().manual_seed(seed * 100 + layer)
        keys.append(torch.randn(KV_HEADS, n_tokens, HEAD_DIM, generator=gen))
        values.append(torch.randn(KV_HEADS, n_tokens, HEAD_DIM, generator=gen))
    return keys, values


def new_pool(num_blocks: int = 16) -> PagedKVCache:
    return PagedKVCache(
        num_blocks=num_blocks,
        block_size=BLOCK,
        num_layers=LAYERS,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
    )


def paged_for(model, *, batch: int = 1, num_blocks: int = 16, block_size: int = BLOCK):
    """A PagedCache shaped like `model`: one empty block table per batch row."""
    config = model.config
    cache = PagedKVCache(
        num_blocks=num_blocks,
        block_size=block_size,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
    )
    allocator = BlockAllocator(num_blocks)
    return PagedCache(cache, allocator, [BlockTable() for _ in range(batch)])


def our_model(model):
    """Swap in our attention; the original model stays untouched as the reference."""
    clone = copy.deepcopy(model)
    swap_attention(clone)
    return clone


# --- the step primitives ----------------------------------------------------


def test_begin_write_commit_writes_the_same_bytes_as_append():
    """The layer-at-a-time path and the all-layers-at-once path are one algorithm."""
    pool = new_pool()
    via_append, via_primitives = BlockTable(), BlockTable()
    keys, values = make_step(3)

    # Separate allocators so both sequences land in the same physical blocks and the
    # comparison is about bytes, not about which ids happened to be handed out.
    pool.append(BlockAllocator(8), via_append, keys, values)
    primitives_allocator = BlockAllocator(8)
    pool.begin_step(primitives_allocator, via_primitives, 3)
    for layer in range(LAYERS):
        pool.write_layer(via_primitives, layer, keys[layer], values[layer])
    pool.commit(via_primitives, 3)

    assert via_primitives.num_tokens == via_append.num_tokens == 3
    assert via_primitives.block_ids == via_append.block_ids
    for layer in range(LAYERS):
        appended_k, appended_v = pool.gather(via_append, layer)
        written_k, written_v = pool.gather(via_primitives, layer)
        assert torch.equal(written_k, appended_k)
        assert torch.equal(written_v, appended_v)


def test_gather_can_include_an_uncommitted_step():
    """Attention must see the token it is processing, before `commit` moves the end."""
    pool = new_pool()
    allocator = BlockAllocator(8)
    table = BlockTable()
    second_keys, second_values = make_step(2, seed=1)
    pool.append(allocator, table, *make_step(1))

    pool.begin_step(allocator, table, 2)
    for layer in range(LAYERS):
        pool.write_layer(table, layer, second_keys[layer], second_values[layer])

    committed_k, _ = pool.gather(table, 0)  # default: committed tokens only
    pending_k, _ = pool.gather(table, 0, length=3)

    assert committed_k.shape[1] == 1
    assert pending_k.shape[1] == 3
    assert torch.equal(pending_k[:, :1], committed_k)  # the old token did not move
    assert torch.equal(pending_k[:, 1:], second_keys[0])


def test_writing_without_reserving_blocks_is_rejected():
    """A block id that was never allocated belongs to somebody else."""
    pool = new_pool()
    keys, _ = make_step(2)

    with pytest.raises(RuntimeError, match="call begin_step first"):
        pool.write_layer(BlockTable(), 0, keys[0], keys[0])


def test_a_step_that_spans_two_blocks_is_split():
    """4 tokens into blocks of 3: the write lands as 3 + 1 in two different blocks."""
    pool = new_pool()
    allocator = BlockAllocator(8)
    table = BlockTable()
    keys, values = make_step(4)

    pool.append(allocator, table, keys, values)

    assert len(table.block_ids) == math.ceil(4 / BLOCK) == 2
    gathered_k, _ = pool.gather(table, 0)
    assert torch.equal(gathered_k, keys[0])


# --- the adapter contract the model relies on -------------------------------


def test_cache_answers_the_questions_the_model_asks():
    cache = PagedCache(new_pool(), BlockAllocator(16), [BlockTable()])
    assert cache.get_seq_length() == 0
    assert cache.get_query_offset() == 0
    assert cache.get_mask_sizes(5) == (5, 0)

    keys, values = make_step(4)
    cache.cache.append(cache.allocator, cache.tables[0], keys, values)

    assert cache.get_seq_length() == 4
    assert cache.get_query_offset() == 4
    assert cache.get_mask_sizes(1) == (5, 0)  # one new token attends to 4 + 1


# --- the real thing: our attention, writing into blocks ---------------------


def test_paged_attention_matches_dynamic_cache_on_prefill(tiny_model):
    """Same weights, same math: only the storage changes, so equality must be exact."""
    ours = our_model(tiny_model)
    cache = paged_for(ours)

    with torch.inference_mode():
        expected = ours(IDS, use_cache=True)  # HF's DynamicCache
        actual = ours(IDS, past_key_values=cache, use_cache=True)

    assert torch.equal(actual.logits, expected.logits)


def test_blocks_hold_exactly_what_the_contiguous_cache_holds(tiny_model):
    """Stronger than matching logits: the stored K/V themselves must be identical."""
    ours = our_model(tiny_model)
    cache = paged_for(ours)

    with torch.inference_mode():
        expected = ours(IDS, use_cache=True).past_key_values
        ours(IDS, past_key_values=cache, use_cache=True)

    for layer in range(tiny_model.config.num_hidden_layers):
        keys, values = cache.cache.gather(cache.tables[0], layer)
        assert torch.equal(keys, expected.layers[layer].keys[0])
        assert torch.equal(values, expected.layers[layer].values[0])


def test_paged_attention_matches_dynamic_cache_while_decoding(tiny_model):
    ours = our_model(tiny_model)
    cache = paged_for(ours)

    with torch.inference_mode():
        reference = ours(IDS, use_cache=True)
        actual = ours(IDS, past_key_values=cache, use_cache=True)
        for step in range(DECODE_STEPS):
            token = reference.logits[:, -1].argmax(-1, keepdim=True)
            reference = ours(token, past_key_values=reference.past_key_values, use_cache=True)
            actual = ours(token, past_key_values=cache, use_cache=True)
            assert torch.equal(actual.logits, reference.logits), f"step {step}"


def test_paged_attention_matches_dynamic_cache_in_a_left_padded_batch(tiny_model):
    """Paging is per sequence; a batch is a loop over block tables."""
    ours = our_model(tiny_model)
    cache = paged_for(ours, batch=2)

    ids = torch.tensor([[0, 0, 1, 2, 3], [4, 5, 6, 7, 8]])
    mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    positions = (mask.cumsum(-1) - 1).clamp(min=0)

    with torch.inference_mode():
        expected = ours(ids, attention_mask=mask, position_ids=positions, use_cache=True)
        actual = ours(
            ids,
            attention_mask=mask,
            position_ids=positions,
            past_key_values=cache,
            use_cache=True,
        )
        assert torch.equal(actual.logits, expected.logits)

        for step in range(2):
            token = expected.logits[:, -1].argmax(-1, keepdim=True)
            mask = torch.cat([mask, mask.new_ones((2, 1))], dim=1)
            positions = (mask.cumsum(-1) - 1)[:, -1:]
            expected = ours(
                token,
                attention_mask=mask,
                position_ids=positions,
                past_key_values=expected.past_key_values,
                use_cache=True,
            )
            actual = ours(
                token,
                attention_mask=mask,
                position_ids=positions,
                past_key_values=cache,
                use_cache=True,
            )
            assert torch.equal(actual.logits, expected.logits), f"step {step}"


def test_paged_generation_picks_the_same_tokens_as_hugging_face(tiny_model):
    """The end-to-end claim: greedy decoding through blocks == HF's greedy decoding."""
    ours = our_model(tiny_model)
    cache = paged_for(ours)

    def greedy(model, past):
        tokens = []
        with torch.inference_mode():
            out = model(IDS, past_key_values=past, use_cache=True)
            for _ in range(5):
                token = out.logits[:, -1].argmax(-1, keepdim=True)
                tokens.append(int(token))
                # The model hands the cache back; with a PagedCache it is the same
                # object we passed in, with HF's DynamicCache it is the grown one.
                out = model(token, past_key_values=out.past_key_values, use_cache=True)
        return tokens

    assert greedy(ours, cache) == greedy(tiny_model, None)


def test_blocks_grow_as_the_sequence_grows(tiny_model):
    ours = our_model(tiny_model)
    cache = paged_for(ours)

    with torch.inference_mode():
        step = ours(IDS, past_key_values=cache, use_cache=True)  # 5 tokens
        assert len(cache.tables[0].block_ids) == 2  # blocks of 3: 4..6 needs a third
        assert cache.length == 5

        for extra in range(2):  # 6 and 7 tokens
            token = step.logits[:, -1].argmax(-1, keepdim=True)
            step = ours(token, past_key_values=cache, use_cache=True)
            assert cache.length == 6 + extra

    assert len(cache.tables[0].block_ids) == 3  # 7 tokens no longer fit in two blocks


def test_freeing_the_adapter_returns_every_block(tiny_model):
    ours = our_model(tiny_model)
    cache = paged_for(ours, batch=2, num_blocks=16)

    with torch.inference_mode():
        ours(
            torch.tensor([[1, 2, 3], [4, 5, 6]]),
            past_key_values=cache,
            use_cache=True,
        )

    assert cache.allocator.num_free < 16  # blocks really were taken
    cache.free()
    assert cache.allocator.num_free == 16  # and every one came back


def test_a_mismatched_batch_size_is_rejected(tiny_model):
    ours = our_model(tiny_model)
    cache = paged_for(ours, batch=2)  # two block tables, one row of input

    with pytest.raises(ValueError, match="block tables"):
        ours(IDS, past_key_values=cache, use_cache=True)
