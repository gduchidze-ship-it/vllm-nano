"""KV cache tests: the contiguous reference, then the paged version that must match it.

Everything here runs on CPU with synthetic tensors except the last test, which
anchors the memory layout to what Hugging Face actually produces.
"""

import math

import pytest
import torch

from vllm_nano.core.kv_cache import (
    BlockAllocator,
    BlockTable,
    KVCache,
    PagedKVCache,
    blocks_for_budget,
)

LAYERS = 2
KV_HEADS = 2
HEAD_DIM = 8
BLOCK = 4


def make_step(n_tokens: int, seed: int = 0) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """One step's K/V, with different numbers per layer so a layer mix-up is visible."""
    keys, values = [], []
    for layer in range(LAYERS):
        gen = torch.Generator().manual_seed(seed * 100 + layer)
        keys.append(torch.randn(KV_HEADS, n_tokens, HEAD_DIM, generator=gen))
        values.append(torch.randn(KV_HEADS, n_tokens, HEAD_DIM, generator=gen))
    return keys, values


def new_contiguous(max_len: int = 16) -> KVCache:
    return KVCache(num_layers=LAYERS, num_kv_heads=KV_HEADS, head_dim=HEAD_DIM, max_len=max_len)


def new_paged(num_blocks: int = 8, block_size: int = BLOCK) -> PagedKVCache:
    return PagedKVCache(
        num_blocks=num_blocks,
        block_size=block_size,
        num_layers=LAYERS,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
    )


# --- contiguous cache -------------------------------------------------------


def test_append_tracks_the_valid_length():
    cache = new_contiguous()
    keys, values = make_step(3)

    cache.append(keys, values)

    assert cache.length == 3
    assert cache.keys(0).shape == (KV_HEADS, 3, HEAD_DIM)
    assert torch.equal(cache.keys(0), keys[0])
    assert torch.equal(cache.values(1), values[1])  # layers do not get swapped


def test_second_append_leaves_the_first_tokens_alone():
    cache = new_contiguous()
    first_keys, first_values = make_step(3)
    cache.append(first_keys, first_values)

    second_keys, second_values = make_step(2, seed=1)
    cache.append(second_keys, second_values)

    assert cache.length == 5
    assert torch.equal(cache.keys(0)[:, :3, :], first_keys[0])
    assert torch.equal(cache.keys(0)[:, 3:, :], second_keys[0])


def test_wrong_number_of_layers_is_rejected_before_writing():
    cache = new_contiguous()
    keys, values = make_step(2)

    with pytest.raises(ValueError, match="expected 2 key and value tensors"):
        cache.append(keys[:1], values)

    assert cache.length == 0  # nothing was half-written


def test_transposed_shape_is_rejected():
    """(n_tokens, kv_heads, head_dim) is the classic mix-up; the cache must say no.

    3 tokens (not 2) so the transposed shape cannot accidentally match kv_heads=2.
    """
    cache = new_contiguous()
    keys, values = make_step(3)

    with pytest.raises(ValueError, match="num_kv_heads=2"):
        cache.append([k.transpose(0, 1) for k in keys], values)


def test_layers_must_advance_by_the_same_amount():
    cache = new_contiguous()
    keys, values = make_step(3)

    with pytest.raises(ValueError, match="same number of tokens"):
        cache.append([keys[0], keys[1][:, :1, :]], values)

    assert cache.length == 0


def test_writing_past_max_len_is_rejected():
    cache = new_contiguous(max_len=4)
    cache.append(*make_step(3))

    with pytest.raises(ValueError, match="cache full"):
        cache.append(*make_step(2))

    assert cache.length == 3


def test_layer_out_of_range_is_rejected():
    cache = new_contiguous()

    with pytest.raises(IndexError, match="out of range"):
        cache.keys(LAYERS)


# --- allocator and block table ---------------------------------------------


def test_allocator_hands_out_each_block_once():
    allocator = BlockAllocator(3)

    handed_out = [allocator.allocate() for _ in range(3)]

    assert sorted(handed_out) == [0, 1, 2]
    assert allocator.num_free == 0
    with pytest.raises(RuntimeError, match="out of KV blocks"):
        allocator.allocate()


def test_freed_block_can_be_reused():
    allocator = BlockAllocator(2)
    first = allocator.allocate()
    allocator.allocate()

    allocator.free(first)

    assert allocator.num_free == 1
    assert allocator.allocate() == first  # the block came back, not a new one


def test_double_free_is_rejected():
    allocator = BlockAllocator(2)
    block_id = allocator.allocate()
    allocator.free(block_id)

    with pytest.raises(ValueError, match="double free"):
        allocator.free(block_id)


def test_freeing_a_block_that_was_never_allocated_is_rejected():
    allocator = BlockAllocator(2)

    with pytest.raises(ValueError, match="not allocated"):
        allocator.free(1)


def test_block_table_maps_positions_to_physical_cells():
    table = BlockTable(block_ids=[7, 2], num_tokens=10)

    assert table.physical_offset(0, BLOCK) == (7, 0)
    assert table.physical_offset(6, BLOCK) == (2, 2)  # second block, third slot

    with pytest.raises(IndexError, match="out of range"):
        table.physical_offset(10, BLOCK)


# --- paged cache ------------------------------------------------------------


def test_table_grows_one_block_per_block_size_tokens():
    allocator = BlockAllocator(8)
    cache = new_paged()
    table = BlockTable()

    for token_count in range(1, 10):
        cache.append(allocator, table, *make_step(1, seed=token_count))
        assert len(table) == math.ceil(token_count / BLOCK)

    assert table.num_tokens == 9


def test_paged_gather_equals_the_contiguous_cache():
    """The invariant for all of step 3: paging changes addresses, not values."""
    cache = new_paged()
    reference = new_contiguous()
    allocator = BlockAllocator(8)
    table = BlockTable()

    for n_tokens in (3, 3, 2):  # 8 tokens, so writes cross block boundaries
        keys, values = make_step(n_tokens, seed=n_tokens)
        cache.append(allocator, table, keys, values)
        reference.append(keys, values)

    for layer in range(LAYERS):
        paged_k, paged_v = cache.gather(table, layer)
        assert torch.equal(paged_k, reference.keys(layer))
        assert torch.equal(paged_v, reference.values(layer))


def test_blocks_are_not_assumed_to_be_physically_contiguous():
    allocator = BlockAllocator(4)
    allocator.allocate()  # 0
    second = allocator.allocate()  # 1
    allocator.allocate()  # 2
    allocator.free(second)  # free queue is now [3, 1]

    table = BlockTable()
    table.block_ids.append(allocator.allocate())  # 3
    table.block_ids.append(allocator.allocate())  # 1
    table.num_tokens = 2 * BLOCK

    # Logical blocks 0 and 1 map to physical blocks 3 and 1: no contiguity anywhere.
    assert table.block_ids == [3, 1]
    assert table.physical_offset(0, BLOCK) == (3, 0)
    assert table.physical_offset(BLOCK, BLOCK) == (1, 0)


def test_gather_follows_a_scattered_block_table():
    """Blocks 4, 5, 1 in that order: still correct, because only the table matters."""
    allocator = BlockAllocator(6)
    allocator.allocate()  # 0
    hole = allocator.allocate()  # 1
    allocator.allocate()  # 2
    allocator.allocate()  # 3
    allocator.free(hole)  # free queue becomes [4, 5, 1]

    cache = new_paged(num_blocks=6)
    reference = new_contiguous()
    table = BlockTable()
    for n_tokens in (BLOCK, BLOCK, BLOCK):
        keys, values = make_step(n_tokens, seed=n_tokens)
        cache.append(allocator, table, keys, values)
        reference.append(keys, values)

    assert table.block_ids == [4, 5, 1]
    for layer in range(LAYERS):
        paged_k, paged_v = cache.gather(table, layer)
        assert torch.equal(paged_k, reference.keys(layer))
        assert torch.equal(paged_v, reference.values(layer))


def test_free_of_a_sequence_returns_all_its_blocks():
    allocator = BlockAllocator(8)
    cache = new_paged()
    table = BlockTable()
    free_before = allocator.num_free

    cache.append(allocator, table, *make_step(5))
    assert allocator.num_free == free_before - math.ceil(5 / BLOCK)

    cache.free(allocator, table)

    assert allocator.num_free == free_before  # zero leaked blocks
    assert table.num_tokens == 0
    assert len(table) == 0


def test_sequence_bigger_than_the_pool_is_rejected():
    allocator = BlockAllocator(1)
    cache = new_paged(num_blocks=1)
    table = BlockTable()

    with pytest.raises(RuntimeError, match="pool has 1"):
        cache.append(allocator, table, *make_step(BLOCK + 1))

    assert table.num_tokens == 0
    assert allocator.num_free == 1


def test_gather_of_an_empty_table_is_empty():
    cache = new_paged()

    keys, values = cache.gather(BlockTable(), 0)

    assert keys.shape == (KV_HEADS, 0, HEAD_DIM)
    assert values.shape == (KV_HEADS, 0, HEAD_DIM)


# --- memory math and HF layout ---------------------------------------------


def test_blocks_for_budget_matches_the_24kb_per_token_formula():
    # Qwen2.5-0.5B in fp32: 24 KB per token, so a 24 MB budget holds 1024 tokens.
    blocks = blocks_for_budget(
        24 * 1024 * 1024,
        block_size=16,
        num_layers=24,
        num_kv_heads=2,
        head_dim=64,
    )

    assert blocks * 16 == 1024


def test_contiguous_cache_reproduces_hf_layout(tiny_model):
    """Our buffer must hold exactly the bytes HF's DynamicCache holds."""
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    config = tiny_model.config
    head_dim = config.hidden_size // config.num_attention_heads

    with torch.inference_mode():
        hf_cache = tiny_model(ids, use_cache=True).past_key_values

    cache = KVCache(
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=head_dim,
        max_len=16,
    )
    cache.append(
        [hf_cache.layers[i].keys[0] for i in range(config.num_hidden_layers)],
        [hf_cache.layers[i].values[0] for i in range(config.num_hidden_layers)],
    )

    for layer in range(config.num_hidden_layers):
        assert torch.equal(cache.keys(layer), hf_cache.layers[layer].keys[0])
        assert torch.equal(cache.values(layer), hf_cache.layers[layer].values[0])
