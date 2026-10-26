"""Drop the paged KV cache into a Hugging Face model without touching the model.

The model only ever asks `past_key_values` four questions:

    get_seq_length()          where does this step start?       (position ids)
    get_query_offset()        same thing, for the mask           (query side)
    get_mask_sizes(q_len)     how wide is the mask?              (kv side)
    update(k, v, layer_idx)   here is this layer's new K/V       (once per layer)

`update` must hand back the *whole* history for that layer, because attention still
multiplies against one contiguous tensor. That is what `PagedKVCache.gather` builds:
the numbers are read out of whatever blocks the table points at, copied, and returned.

So this class is a translator, not a cache. The storage and the block bookkeeping
live in `core/kv_cache.py`; the per-sequence block tables live with whoever schedules
requests (one table per row here, and one per request once there is a scheduler).

What it does *not* do yet: avoid the gather. A real PagedAttention kernel computes
`q @ K_block.T` block by block and never materializes the history. That is the next
step, and this class is where the reading happens, so it is where that change lands.
"""

from __future__ import annotations

import torch

from vllm_nano.core.kv_cache import BlockAllocator, BlockTable, PagedKVCache


class PagedCache:
    """A view over a shared paged pool, shaped like Hugging Face's cache.

    Holds one `BlockTable` per batch row, and translates the four cache calls above
    into block allocation, per-layer writes, and a gather. All rows advance together
    -- which is true for a batch of sequences decoded in lockstep, including
    left-padded rows whose pad tokens are stored and then masked out by attention.
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
        """Store this layer's K/V in blocks, then return the whole history.

        `key_states`/`value_states` are (batch, kv_heads, n_tokens, head_dim). The
        batch axis is dropped by writing each row into its own table: paging is per
        sequence, batching is a loop over sequences.

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

        # The step's tokens are written but not committed, so pass the length
        # explicitly -- otherwise attention would not see the token it is processing.
        length = self.tables[0].num_tokens + self._pending
        gathered = [self.cache.gather(table, layer_idx, length=length) for table in self.tables]
        # Per row, then a stack: for B>1 this copies the history twice. Both copies
        # are on the list of things the block-wise kernel deletes.
        keys = torch.stack([k for k, _ in gathered])
        values = torch.stack([v for _, v in gathered])

        if layer_idx == self.cache.num_layers - 1:
            for table in self.tables:
                self.cache.commit(table, self._pending)
            self._pending = 0

        return keys, values

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
