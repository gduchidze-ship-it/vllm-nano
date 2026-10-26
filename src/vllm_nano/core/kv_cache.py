"""KV cache: store each token's K/V once, then read it back on every step.

Two implementations, deliberately in this order:

* `KVCache`      -- one contiguous buffer per layer, sized to `max_len`. Simple, and
                    the reference the paged version has to match exactly.
* `PagedKVCache` -- a pool of fixed-size blocks plus a per-sequence `BlockTable`.
                    A sequence's K/V can live in blocks scattered anywhere in the
                    pool, so it grows without copying and without reserving `max_len`
                    for every request. A step is written with `begin_step` (allocate),
                    `write_layer` (once per layer) and `commit` (advance the length),
                    because a transformer hands attention one layer at a time.

Shapes (one sequence, no batch dimension -- paging is inherently per sequence):

    one token of one layer -> (num_kv_heads, head_dim)
    a step                 -> (num_kv_heads, n_tokens, head_dim)
    the whole cache        -> (num_kv_heads, length, head_dim)

`seq_len` sits *before* `head_dim`, matching Hugging Face's
(batch, kv_heads, seq_len, head_dim) with the batch axis dropped. Slicing the wrong
axis is the classic bug in this file, so every write takes an explicit token range.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import torch


def _check_layer_tensor(tensor: torch.Tensor, *, num_kv_heads: int, head_dim: int) -> int:
    """Validate one layer's slice of a step and return how many tokens it carries.

    Shape must be (num_kv_heads, n_tokens, head_dim). Checking this *before* writing
    matters: a transposed tensor silently stores the right number of numbers in the
    wrong places, which no later assertion can untangle.
    """
    if tensor.ndim != 3 or tensor.shape[0] != num_kv_heads or tensor.shape[2] != head_dim:
        raise ValueError(
            f"expected (num_kv_heads={num_kv_heads}, n_tokens, head_dim={head_dim}), "
            f"got {tuple(tensor.shape)}"
        )
    return tensor.shape[1]


def _check_step(
    keys: list[torch.Tensor],
    values: list[torch.Tensor],
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
) -> int:
    """Validate one step's K/V and return how many tokens it carries.

    Called *before* anything is written, so a bad step leaves the cache untouched:
    a half-written step would corrupt every later offset.
    """
    if len(keys) != num_layers or len(values) != num_layers:
        raise ValueError(
            f"expected {num_layers} key and value tensors, got {len(keys)} and {len(values)}"
        )

    n_tokens: int | None = None
    for tensor in (*keys, *values):
        layer_tokens = _check_layer_tensor(tensor, num_kv_heads=num_kv_heads, head_dim=head_dim)
        if n_tokens is None:
            n_tokens = layer_tokens
        elif layer_tokens != n_tokens:
            raise ValueError("every layer must advance by the same number of tokens")
    return n_tokens if n_tokens is not None else 0


def blocks_for_budget(
    budget_bytes: int,
    *,
    block_size: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.float32,
) -> int:
    """How many blocks fit in a byte budget.

    Same formula as NOTES.md: 2 (K and V) x layers x kv_heads x head_dim x bytes
    per token. Qwen2.5-0.5B in fp32 = 24 KB/token, so 24 MB buys 1024 tokens.
    """
    bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * torch.finfo(dtype).bits // 8
    return budget_bytes // (block_size * bytes_per_token)


class KVCache:
    """Contiguous KV cache for one sequence, preallocated to `max_len` tokens.

    This is the straightforward version: token `i` of layer `l` lives at
    `buffer[l][:, i, :]`. No indirection, no allocator -- and no way to share
    memory between two sequences of different lengths.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_len: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> None:
        if min(num_layers, num_kv_heads, head_dim, max_len) < 1:
            raise ValueError("num_layers, num_kv_heads, head_dim and max_len must be positive")

        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_len = max_len

        # `empty`, not `zeros`: reading past `length` then yields garbage instead of a
        # convincing row of zeros. That bug is invisible with zeros and obvious here.
        shape = (num_kv_heads, max_len, head_dim)
        self._k = [torch.empty(shape, dtype=dtype, device=device) for _ in range(num_layers)]
        self._v = [torch.empty(shape, dtype=dtype, device=device) for _ in range(num_layers)]
        self._length = 0

    @property
    def length(self) -> int:
        """How many tokens are valid. Always <= max_len, the buffer's real size."""
        return self._length

    def append(self, keys: list[torch.Tensor], values: list[torch.Tensor]) -> None:
        """Write one step (all layers at once) at the current end of the cache."""
        n_tokens = _check_step(
            keys,
            values,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
        )
        if self._length + n_tokens > self.max_len:
            raise ValueError(
                f"cache full: {self._length} valid + {n_tokens} new > max_len {self.max_len}"
            )

        for layer in range(self.num_layers):
            # Every layer writes at the same offset, so `_length` advances once, after
            # the loop. Bumping it inside the loop would shift layers 1..n by one step.
            window = slice(self._length, self._length + n_tokens)
            self._k[layer][:, window, :] = keys[layer]
            self._v[layer][:, window, :] = values[layer]
        self._length += n_tokens

    def keys(self, layer: int) -> torch.Tensor:
        """Valid K for one layer: (num_kv_heads, length, head_dim). A view, not a copy."""
        return self._k[self._check_layer(layer)][:, : self._length, :]

    def values(self, layer: int) -> torch.Tensor:
        """Valid V for one layer: (num_kv_heads, length, head_dim)."""
        return self._v[self._check_layer(layer)][:, : self._length, :]

    def _check_layer(self, layer: int) -> int:
        if not 0 <= layer < self.num_layers:
            raise IndexError(f"layer {layer} out of range 0..{self.num_layers - 1}")
        return layer


class BlockAllocator:
    """Hands out physical block ids and takes them back.

    The pool itself is just a big tensor; this class is the only mutable state that
    says which rows of it are free. Freed blocks are reused last-in-first-out, which
    keeps a freshly freed block out of circulation for a while (fewer aliasing bugs).
    """

    def __init__(self, num_blocks: int) -> None:
        if num_blocks < 1:
            raise ValueError("num_blocks must be positive")
        self.num_blocks = num_blocks
        self._free = deque(range(num_blocks))
        self._is_free = set(range(num_blocks))  # membership check for double free

    @property
    def num_free(self) -> int:
        return len(self._free)

    def allocate(self) -> int:
        if not self._free:
            raise RuntimeError(
                f"out of KV blocks: all {self.num_blocks} are in use. "
                "A real engine would preempt or queue the request here."
            )
        block_id = self._free.popleft()
        self._is_free.remove(block_id)
        return block_id

    def free(self, block_id: int) -> None:
        # A double free is a leak-sized bug: the same block would be handed to two
        # sequences and both would silently corrupt each other.
        if not 0 <= block_id < self.num_blocks:
            raise ValueError(f"block {block_id} out of range 0..{self.num_blocks - 1}")
        if block_id in self._is_free:
            raise ValueError(f"block {block_id} is not allocated (double free?)")
        self._is_free.add(block_id)
        self._free.append(block_id)


@dataclass
class BlockTable:
    """One sequence's map from logical position to physical block id.

    `block_ids[i]` holds logical positions [i * block_size, (i + 1) * block_size).
    `num_tokens` is how many positions are valid; the last block is usually partly
    filled, which is the only waste paging has.
    """

    block_ids: list[int] = field(default_factory=list)
    num_tokens: int = 0

    def __len__(self) -> int:
        return len(self.block_ids)

    def physical_offset(self, position: int, block_size: int) -> tuple[int, int]:
        """Map a logical token position to (physical block, offset inside that block)."""
        if not 0 <= position < self.num_tokens:
            raise IndexError(f"position {position} out of range 0..{self.num_tokens - 1}")
        return self.block_ids[position // block_size], position % block_size


def _write_segments(
    table: BlockTable, start: int, n_tokens: int, block_size: int
) -> list[tuple[int, int, int, int]]:
    """Split a write of `n_tokens` into pieces that each fit inside one block.

    Returns (physical_block, offset_in_block, take, src_start) tuples. A step that
    spans a block boundary produces two pieces -- that is the whole point of paging.
    """
    segments: list[tuple[int, int, int, int]] = []
    position, remaining, src_start = start, n_tokens, 0
    while remaining:
        offset = position % block_size
        take = min(block_size - offset, remaining)
        segments.append((table.block_ids[position // block_size], offset, take, src_start))
        position += take
        src_start += take
        remaining -= take
    return segments


class PagedKVCache:
    """Fixed-size block pool; K/V live wherever the block tables say they live.

    One big tensor for all sequences instead of one buffer per sequence:

        (num_blocks, num_layers, num_kv_heads, block_size, head_dim)

    The last two axes are contiguous, which is what an attention kernel wants to read.
    """

    def __init__(
        self,
        *,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> None:
        if min(num_blocks, block_size, num_layers, num_kv_heads, head_dim) < 1:
            raise ValueError("all sizes must be positive")

        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        shape = (num_blocks, num_layers, num_kv_heads, block_size, head_dim)
        self._k = torch.empty(shape, dtype=dtype, device=device)
        self._v = torch.empty(shape, dtype=dtype, device=device)

    @property
    def bytes_per_token(self) -> int:
        element = self._k.element_size()
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * element

    def append(
        self,
        allocator: BlockAllocator,
        table: BlockTable,
        keys: list[torch.Tensor],
        values: list[torch.Tensor],
    ) -> None:
        """Write one step for one sequence, allocating blocks as the sequence grows.

        Convenience wrapper around begin/write/commit for callers that have every
        layer in hand at once (tests, offline replay). A transformer does not: it
        calls attention one layer at a time, which is why the three primitives exist.
        """
        n_tokens = _check_step(
            keys,
            values,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
        )
        self.begin_step(allocator, table, n_tokens)
        for layer in range(self.num_layers):
            self.write_layer(table, layer, keys[layer], values[layer])
        self.commit(table, n_tokens)

    def begin_step(self, allocator: BlockAllocator, table: BlockTable, n_tokens: int) -> None:
        """Reserve space for `n_tokens` more tokens, without advancing the length.

        Reserving and advancing are separate because a step is written one layer at
        a time. Every layer must write at the same offset, so the offset can only move
        after the last layer has written -- that is `commit`. Advance inside the layer
        loop and layer 1 lands one step later than layer 0, forever.
        """
        if n_tokens < 0:
            raise ValueError(f"n_tokens must not be negative, got {n_tokens}")
        self._grow(allocator, table, table.num_tokens + n_tokens)

    def write_layer(
        self, table: BlockTable, layer: int, keys: torch.Tensor, values: torch.Tensor
    ) -> None:
        """Write one layer's slice of the current step, still without advancing.

        `keys`/`values` are that layer only: (num_kv_heads, n_tokens, head_dim).
        """
        self._check_layer(layer)
        for tensor in (keys, values):
            # Writing bf16 into an fp32 pool "works" (a silent cast) and then breaks
            # where it is hard to see: gather hands attention an fp32 history.
            if tensor.dtype != self._k.dtype or tensor.device != self._k.device:
                raise ValueError(
                    f"pool is {self._k.dtype} on {self._k.device}, but this step is "
                    f"{tensor.dtype} on {tensor.device}; build the pool to match the model"
                )
        n_keys = _check_layer_tensor(keys, num_kv_heads=self.num_kv_heads, head_dim=self.head_dim)
        n_values = _check_layer_tensor(
            values, num_kv_heads=self.num_kv_heads, head_dim=self.head_dim
        )
        if n_keys != n_values:
            raise ValueError(f"keys carry {n_keys} tokens, values carry {n_values}")
        # Blocks must already exist: writing into a block id that was never handed out
        # would go to another sequence's memory.
        capacity = len(table.block_ids) * self.block_size
        if table.num_tokens + n_keys > capacity:
            raise RuntimeError(
                f"writing {n_keys} tokens at {table.num_tokens} needs "
                f"{math.ceil((table.num_tokens + n_keys) / self.block_size)} blocks, "
                f"but only {len(table.block_ids)} were reserved; call begin_step first"
            )

        for block_id, offset, take, src_start in _write_segments(
            table, table.num_tokens, n_keys, self.block_size
        ):
            window = slice(offset, offset + take)
            source = slice(src_start, src_start + take)
            self._k[block_id, layer, :, window, :] = keys[:, source, :]
            self._v[block_id, layer, :, window, :] = values[:, source, :]

    def commit(self, table: BlockTable, n_tokens: int) -> None:
        """Advance the sequence past the `n_tokens` just written.

        One call per step, after every layer. From here on those tokens are part of
        the cache and show up in `gather` and in the mask's length.
        """
        if n_tokens < 0:
            raise ValueError(f"n_tokens must not be negative, got {n_tokens}")
        capacity = len(table.block_ids) * self.block_size
        if table.num_tokens + n_tokens > capacity:
            raise RuntimeError(
                f"committing {n_tokens} tokens at {table.num_tokens} would pass the "
                f"{capacity} reserved slots; call begin_step first"
            )
        table.num_tokens += n_tokens

    def _grow(self, allocator: BlockAllocator, table: BlockTable, total_tokens: int) -> None:
        """Make sure the table has enough blocks for `total_tokens` tokens."""
        needed_blocks = math.ceil(total_tokens / self.block_size)
        if needed_blocks > self.num_blocks:
            raise RuntimeError(
                f"sequence needs {needed_blocks} blocks but the pool has {self.num_blocks}"
            )

        # Growth is one id appended per new block, never a copy of past tokens.
        while len(table.block_ids) < needed_blocks:
            table.block_ids.append(allocator.allocate())

    def gather(
        self, table: BlockTable, layer: int, length: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Collect one layer's blocks into a contiguous (kv_heads, length, head_dim).

        `length` defaults to the committed length. Pass it explicitly to include the
        tokens of a step that has been written but not yet committed -- which is
        exactly what attention needs: it must see the token it is processing.

        This copy is the cost PagedAttention removes: a real kernel attends block by
        block instead of materializing the contiguous view. Correct first, fast later.
        """
        self._check_layer(layer)
        length = table.num_tokens if length is None else length
        if length < 0:
            raise ValueError(f"length must not be negative, got {length}")
        if length > len(table.block_ids) * self.block_size:
            raise RuntimeError(
                f"asked for {length} tokens but only {len(table.block_ids)} blocks "
                f"are reserved ({len(table.block_ids) * self.block_size} slots)"
            )
        if length == 0:
            empty = torch.empty(
                self.num_kv_heads, 0, self.head_dim, dtype=self._k.dtype, device=self._k.device
            )
            return empty, empty.clone()

        used_blocks = math.ceil(length / self.block_size)
        block_ids = table.block_ids[:used_blocks]
        # (block_size) is dim 1 of one block's slice, so cat along dim=1 joins positions.
        k = torch.cat([self._k[block_id, layer] for block_id in block_ids], dim=1)
        v = torch.cat([self._v[block_id, layer] for block_id in block_ids], dim=1)
        return k[:, :length, :], v[:, :length, :]

    def _check_layer(self, layer: int) -> None:
        if not 0 <= layer < self.num_layers:
            raise IndexError(f"layer {layer} out of range 0..{self.num_layers - 1}")

    def free(self, allocator: BlockAllocator, table: BlockTable) -> None:
        """Return every block to the allocator. Forgetting this call is a leak."""
        for block_id in table.block_ids:
            allocator.free(block_id)
        table.block_ids.clear()
        table.num_tokens = 0
