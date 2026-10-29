"""Paged attention: read K/V block by block, never materialize the history.

Two different claims, with two different kinds of proof:

* **Storage is exact.** A block write is a copy, so what sits in the pool must be
  bit-identical to what a contiguous cache holds. `torch.equal`, no tolerance.
* **Attention is mathematically identical, not bit-identical.** Block-wise attention
  merges each block into a running softmax, which adds the same numbers in a different
  order. Floating-point addition is not associative, so expect ~1e-7 in fp32. The real
  gate is the token: greedy decoding must still match Hugging Face exactly.
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
ATOL = 1e-6  # measured ~9e-8; this is the reordering bill, not a model difference


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


def test_a_pool_with_the_wrong_dtype_is_rejected():
    """The pool must match the model; a silent cast would surface far from here."""
    pool = PagedKVCache(
        num_blocks=4,
        block_size=BLOCK,
        num_layers=LAYERS,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=torch.float16,
    )

    with pytest.raises(ValueError, match="build the pool to match the model"):
        pool.append(BlockAllocator(4), BlockTable(), *make_step(2))


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


def test_block_is_a_view_not_a_copy():
    """If reading a block copied it, the whole point of paging would be gone.

    Same memory address means attention multiplies directly against the pool.
    """
    pool = new_pool()
    allocator = BlockAllocator(8)
    table = BlockTable()
    pool.append(allocator, table, *make_step(BLOCK))

    keys, values = pool.block(table, 0, 0)
    physical = table.block_ids[0]
    assert keys.data_ptr() == pool._k[physical, 0].data_ptr()
    assert values.data_ptr() == pool._v[physical, 0].data_ptr()
    assert keys.shape == (KV_HEADS, BLOCK, HEAD_DIM)


def test_a_block_that_was_never_allocated_is_rejected():
    pool = new_pool()

    with pytest.raises(IndexError, match="out of range"):
        pool.block(BlockTable(), 0, 0)


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
    assert cache.step_length() == 4  # nothing pending: committed and readable agree
    assert cache.block_size == BLOCK
    assert cache.block(0, 0)[0].shape == (1, KV_HEADS, BLOCK, HEAD_DIM)


# --- the real thing: our attention, reading blocks --------------------------


@pytest.mark.parametrize("block_size", [1, 2, 3, 7])
def test_paged_attention_matches_contiguous_attention_on_prefill(tiny_model, block_size):
    """Same weights, same math: only the summation order changes.

    The sweep covers the awkward shapes: one token per block, a block boundary
    landing mid-prompt, and a prompt that fits in a single partial block.
    """
    ours = our_model(tiny_model)
    cache = paged_for(ours, block_size=block_size, num_blocks=16)

    with torch.inference_mode():
        expected = ours(IDS, use_cache=True)  # HF's DynamicCache
        actual = ours(IDS, past_key_values=cache, use_cache=True)

    assert torch.allclose(actual.logits, expected.logits, atol=ATOL)
    assert torch.equal(actual.logits.argmax(-1), expected.logits.argmax(-1))
    assert len(cache.tables[0].block_ids) == math.ceil(IDS.shape[1] / block_size)


def test_blocks_hold_exactly_what_the_contiguous_cache_holds(tiny_model):
    """The write itself is a copy, so it must be bit-identical.

    Layer 0 is the clean check: its K/V come straight from the embeddings. Later
    layers can drift in the last bits because their *inputs* come from attention,
    and attention now adds in a different order -- that drift is upstream of the
    cache, not in it.
    """
    ours = our_model(tiny_model)
    cache = paged_for(ours)

    with torch.inference_mode():
        expected = ours(IDS, use_cache=True).past_key_values
        ours(IDS, past_key_values=cache, use_cache=True)

    for layer in range(tiny_model.config.num_hidden_layers):
        keys, values = cache.cache.gather(cache.tables[0], layer)
        if layer == 0:
            assert torch.equal(keys, expected.layers[layer].keys[0])
            assert torch.equal(values, expected.layers[layer].values[0])
        else:
            assert torch.allclose(keys, expected.layers[layer].keys[0], atol=ATOL)
            assert torch.allclose(values, expected.layers[layer].values[0], atol=ATOL)


def test_paged_attention_matches_contiguous_attention_while_decoding(tiny_model):
    ours = our_model(tiny_model)
    cache = paged_for(ours)

    with torch.inference_mode():
        reference = ours(IDS, use_cache=True)
        actual = ours(IDS, past_key_values=cache, use_cache=True)
        for step in range(DECODE_STEPS):
            token = reference.logits[:, -1].argmax(-1, keepdim=True)
            reference = ours(token, past_key_values=reference.past_key_values, use_cache=True)
            actual = ours(token, past_key_values=cache, use_cache=True)
            assert torch.allclose(actual.logits, reference.logits, atol=ATOL), f"step {step}"
            assert torch.equal(actual.logits.argmax(-1), reference.logits.argmax(-1)), (
                f"step {step}"
            )


def test_paged_attention_matches_contiguous_attention_in_a_left_padded_batch(tiny_model):
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
        assert torch.allclose(actual.logits, expected.logits, atol=ATOL)

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
            assert torch.allclose(actual.logits, expected.logits, atol=ATOL), f"step {step}"

    # Pad positions see nothing at all. The running softmax must degrade to a plain
    # average there, not to 0/0.
    assert torch.isfinite(actual.logits).all()


def test_paged_attention_never_gathers_the_history(tiny_model, monkeypatch):
    """The architectural claim, pinned: no block is ever concatenated into a history.

    If someone reintroduces a gather for convenience, this fails immediately -- which
    is the only way to notice, because the output would still be correct.
    """
    ours = our_model(tiny_model)
    cache = paged_for(ours)

    def refuse(*args, **kwargs):
        raise AssertionError("paged attention must read blocks, not gather the history")

    monkeypatch.setattr(PagedKVCache, "gather", refuse)
    with torch.inference_mode():
        logits = ours(IDS, past_key_values=cache, use_cache=True).logits

    assert logits.shape == (1, IDS.shape[1], tiny_model.config.vocab_size)


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
