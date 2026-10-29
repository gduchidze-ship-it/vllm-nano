"""Turn the paged KV cache into something a Hugging Face model can drive.

The model only ever asks `past_key_values` four questions:

    get_seq_length()          where does this step start?       (position ids)
    get_query_offset()        same thing, for the mask           (query side)
    get_mask_sizes(q_len)     how wide is the mask?              (kv side)
    update(k, v, layer_idx)   here is this layer's new K/V       (once per layer)

The first three are bookkeeping and look exactly like a contiguous cache. The fourth
is where paging shows up: `update` stores the step in blocks and hands back the slice
it was given, because the history is *not* gathered into one tensor. Attention reads
it block by block out of the pool instead (`block()`), which is what removes the
sequence-length copy from every layer of every step.

So this class is a translator, not a cache. The storage and the block bookkeeping
live in `core/kv_cache.py`; the per-sequence block tables live with whoever schedules
requests (one table per row here, and one per request once there is a scheduler).
"""

from __future__ import annotations

import torch

from vllm_nano.core.kv_cache import BlockAllocator, BlockTable, PagedKVCache


class PagedCache:
    """A view over a shared paged pool, shaped like Hugging Face's cache.

    Holds one `BlockTable` per batch row, and translates the four cache calls above
    into block allocation, per-layer writes, and block reads for attention. All rows
    advance together -- which is true for a batch of sequences decoded in lockstep,
    including left-padded rows whose pad tokens are stored and then masked out.
    """

    def __init__(
        self,
        cache: PagedKVCache,
        allocator: BlockAllocator,
        tables: list[BlockTable],
    ) -> None:
        if not tables:
            raise ValueError("a paged cache needs at least one block table")
        self.cache = cache
        self.allocator = allocator
        self.tables = tables
        # Tokens written by the layers already visited in the current step, not yet
        # committed. They are visible to attention but do not move `num_tokens` until
        # the last layer has written.
        self._pending = 0

    # --- what the model asks before any layer runs ---------------------------

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Tokens already cached: the position this step starts at.

        Every row holds the same number of tokens (batched decoding advances them
        together), so the first table answers for all of them.
        """
        return self.tables[0].num_tokens

    def get_query_offset(self, layer_idx: int = 0) -> int:
        """Where the queries start in the mask. Same value as the cache length."""
        return self.get_seq_length(layer_idx)

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0) -> tuple[int, int]:
        """(how many key positions the mask must cover, offset of the first one).

        `query_length` new tokens attend to the whole cache plus themselves, so the
        key side is `past + new`. Nothing is cropped, so the offset is zero.
        """
        return self.get_seq_length(layer_idx) + query_length, 0

    # --- what the model asks once per layer ---------------------------------

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args: object,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Store this layer's K/V in blocks and return the step's own tensors.

        `key_states`/`value_states` are (batch, kv_heads, n_tokens, head_dim). The
        batch axis disappears into one write per row: paging is per sequence, batching
        is a loop over sequences.

        The return value intentionally is **not** the history. A contiguous cache has
        to hand attention the whole past because that is the only copy that exists;
        here the blocks are the copy, and attention takes them from `block()` without
        concatenating anything. Returning the step unchanged keeps the signature the
        decoder layer expects while making it obvious there is nothing to unpack.

        Layer 0 starts the step and the last layer ends it. That ordering is what
        Hugging Face's decoder does (layers run 0..N-1 once per forward), and it is
        the reason a step can be written layer by layer at all.
        """
        n_tokens = key_states.shape[2]
        if key_states.shape[0] != len(self.tables):
            raise ValueError(
                f"cache has {len(self.tables)} block tables but K/V has "
                f"batch size {key_states.shape[0]}"
            )

        if layer_idx == 0:
            if self._pending:
                raise RuntimeError(
                    f"layer 0 started a new step while {self._pending} tokens were "
                    "still uncommitted; the previous step never reached the last layer"
                )
            for table in self.tables:
                self.cache.begin_step(self.allocator, table, n_tokens)
            self._pending = n_tokens
        elif n_tokens != self._pending:
            raise RuntimeError(
                f"layer {layer_idx} writes {n_tokens} tokens but the step began with "
                f"{self._pending}; every layer must advance by the same amount"
            )

        for row, table in enumerate(self.tables):
            self.cache.write_layer(table, layer_idx, key_states[row], value_states[row])

        if layer_idx == self.cache.num_layers - 1:
            for table in self.tables:
                self.cache.commit(table, self._pending)
            self._pending = 0

        return key_states, value_states

    # --- what attention reads ------------------------------------------------

    @property
    def block_size(self) -> int:
        """Tokens per physical block. Attention slices its mask by this."""
        return self.cache.block_size

    def step_length(self) -> int:
        """Tokens attention must read now: committed plus the step being written.

        The step's K/V is already in its blocks (that is what `update` just did), so
        it must be included -- otherwise every token would attend to a past that
        stops just short of itself.
        """
        return self.tables[0].num_tokens + self._pending

    def block(self, layer_idx: int, block_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Logical block `block_index`, stacked across rows.

        Returns (batch, kv_heads, block_size, head_dim) for K and for V, assembled
        from that row's own physical block. The stack is a copy of *one block*, not
        of the history: its size is constant no matter how long the sequence gets.
        """
        pairs = [self.cache.block(table, layer_idx, block_index) for table in self.tables]
        return (
            torch.stack([keys for keys, _ in pairs]),
            torch.stack([values for _, values in pairs]),
        )

    # --- housekeeping --------------------------------------------------------

    @property
    def batch_size(self) -> int:
        return len(self.tables)

    @property
    def length(self) -> int:
        """Committed tokens per row. Does not include a step still in flight."""
        return self.tables[0].num_tokens

    def free(self) -> None:
        """Give every block back to the pool. Not freeing is how a pool leaks."""
        for table in self.tables:
            self.cache.free(self.allocator, table)
        self._pending = 0
