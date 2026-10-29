# Notes

## Big picture
- An LLM writes text **one token at a time**; each new token needs all earlier tokens.
- Recomputing everything each step is O(n²) work. The **KV cache** stores the K/V of past tokens so each step only processes the newest token.
- vLLM's other ideas build on that: batching many requests, paging the cache into blocks, a scheduler to decide who runs.
- Correctness check at every step: greedy output must match HF `generate(do_sample=False)` token for token.

## Project setup
- `src/` layout + `[build-system]` in `pyproject.toml`, so `import vllm_nano` works from tests and scripts.
- `torch` and `transformers` are runtime deps; `pytest` and `ruff` are dev-only.
- One module per job (config, sequence, engine, model runner, cache, sampling) so later steps change few files.

## ModelConfig
- Frozen dataclass: shared by the whole engine, so nothing can change it by accident. Copy with `dataclasses.replace`.
- Frozen means no `self.x = ...` in `__post_init__`; use `object.__setattr__` once to set derived values.
- Architecture facts are **read from the HF config**, not hardcoded, so any HF model works.
- Qwen2.5-0.5B: 24 layers, 14 attention heads, **2 KV heads** (GQA), head dim 64.
  Each KV head is shared by 14 / 2 = 7 query heads, which makes the KV cache 7x smaller.

## float32 vs 16-bit
- float32 = 4 bytes, ~7 digits. fp16 / bf16 = 2 bytes. bf16 keeps fp32's range with fewer digits. fp16 overflows above ~65k.
- Weights memory = params x bytes. Qwen 0.5B: ~2 GB in fp32, ~1 GB in 16-bit.
- **KV cache per token** = 2 x layers x kv_heads x head_dim x bytes.
  Qwen: 2 x 24 x 2 x 64 x 4 = **24 KB in fp32**, 12 KB in 16-bit. This number sets how many sequences fit in memory (step 3).
- Using fp32 for now: tiny rounding differences can flip the `argmax` when two logits are nearly tied, so 16-bit would break the exact HF match and hide real bugs. Move to bf16 only after everything is correct.
- Watch out: `mps` has patchy 16-bit support; mixing dtypes causes errors; batching/padding changes the order of float sums, so results can differ slightly.

## Logging
- Library code uses `from loguru import logger`; only the entry point calls `setup_logging()` (once, no side effects on import).
- `request_context(id)` tags every log line with a request id. Needed once many requests run at the same time.
- Secrets (Bearer / `hf_` tokens) are scrubbed from messages and `extra`, not from tracebacks.
- JSON file in `logs/` (git-ignored) for analysis; Rich console for humans.

## KV cache (roadmap step 1)
- One method, two modes: `forward(ids, cache=None)`. `cache is None` = **prefill** (whole prompt, cache 0 -> L). Every later call passes **one** token and the cache grows by 1 = **decode**.
- Why it is legal: causal attention means the logits at position t depend only on tokens <= t. So cached decode must reproduce one full forward at the same position. Test: tiny model, `torch.allclose(..., atol=1e-6)`; measured max diff ~3e-8 in fp32.
- Work per token: prefill is O(L) once. Each decode is O(1) in sequence length for the matmuls (attention still reads the L cached K/V, but nothing is recomputed). Before: O(n) work per step, O(n^2) total.
- No `attention_mask` needed for a single unpadded sequence: transformers derives `cache_position` from the cache length. We need it as soon as a batch is padded (step 2).
- `DynamicCache` grows **in place** and the model returns the same object. Always use the returned cache; a reference you kept "before" now reports the new length.
- Do NOT send the whole sequence with a warm cache: the prompt is already in it, so every token is attended twice and the logits are silently wrong. There is a test that asserts the two differ.
- Memory: 2 (K and V) x layers x kv_heads x head_dim x bytes per token = **24 KB fp32** for Qwen 0.5B (24 MB at 1k, 96 MB at 4k, 768 MB at 32k). Context length costs memory linearly; paging (step 3) is how vLLM spends it in blocks instead of per-sequence slabs.
- The engine now runs one forward per generated token: 1 prefill + (n-1) decodes, and EOS breaks before the decode nobody would read.

## Batching (roadmap step 2)
- One forward per step now serves the whole batch: logits are (batch, vocab) and the engine samples one token per row.
- **Left padding only.** Real sequences end flush against the right edge, so `logits[:, -1]` is the next-token position for every row. Right padding would make that column a pad.
- `attention_mask` must cover **cache + new token** (width grows by one per step). A stale, unpadded mask is accepted silently and gives wrong logits (~0.05 difference on the tiny model) — there is a test for it.
- `position_ids = (mask.cumsum(-1) - 1).clamp(min=0)`: pads must not shift a sequence's positions.
  - Measured surprise: passing `arange` instead does **not** break Qwen. RoPE is relative, and left padding shifts a row's tokens *uniformly*, which cancels out. It matters for absolute-position and ALiBi models, so we pass it explicitly anyway — it is free and model-agnostic.
- Static batching: finished rows keep their slot and are fed `pad_token_id` until the batch ends; their output is ignored. Simple, and obviously wasteful — that waste is what continuous batching (step 4) removes.
- Padding also allocates cache for tokens that do not exist, and the padded width is the longest prompt in the batch. That is the concrete motivation for paging (step 3).
- `cache` is keyword-only in `forward`, because `forward(ids, cache)` silently bound the cache to `attention_mask` and failed with a confusing `.to` error.

## Our own attention (step 3, rung 1)
- `models/attention.py` is a readable copy of Qwen2 attention: proj -> `(b, s, h, d)` -> `(b, h, s, d)` -> RoPE -> cache update -> `q k^T * scale + mask` -> softmax in fp32 -> `@ v` -> `o_proj`.
- Verified against HF: prefill and 4 decode steps agree within **3e-8**, cache contents match, and `generate()` picks the same tokens.
- **The mask is built by the model, not by attention.** With `_attn_implementation="sdpa"` the model passes `attention_mask=None` and expects the kernel to be causal. Our eager matmuls then attend in *both* directions: logits drifted by 0.21 and the output still looked like text. `swap_attention()` sets the flag to `"eager"`; `test_attention_is_causal_even_after_the_swap` guards it (position 0 must not change when later tokens change).
- GQA: K/V are repeated `num_attention_heads // num_kv_heads` times before the matmul. A kernel indexes heads instead of copying.
- The seam for paging *looked* like a single statement (`key, value = past_key_values.update(...)`). That held while the paged cache still gathered a contiguous history (rung 2) and stops holding once it doesn't (rung 3).

## Paging in the attention loop (step 3, rung 2)
- The model never touches the cache directly. It asks `past_key_values` four questions -- `get_seq_length()`, `get_query_offset()`, `get_mask_sizes(q_len)` and once per layer `update(k, v, layer_idx)` -- so anything that answers them *is* a cache. `models/paged_cache.py` is that translator; no model code changed.
- A step is **two-phase** because attention runs one layer at a time: `begin_step` (allocate blocks), `write_layer` x layers, `commit` (advance the length). Advance inside the layer loop and layer 1 writes one step later than layer 0 -- the same bug the contiguous cache documents.
- Mid-step the blocks hold tokens that are written but not committed, so `step_length()` is `committed + pending`. Attention must read the token it is processing.
- Layer 0 opens a step and the last layer closes it. That is an assumption about the decoder loop (layers run 0..N-1 exactly once per forward), written down in `PagedCache.update` and enforced by a counter check.
- Batching is a loop over block tables: one table per row, `key_states[row]` written into `tables[row]`. Paging is per sequence; the batch is the outer loop.
- Left-padded rows: logits at **pad** positions differ from HF by ~0.11, because when every key is masked both sides produce arbitrary junk and produce it differently. Harmless (no real token attends to a pad), but it means comparing against HF only makes sense at real positions -- compare the full tensor against our own `DynamicCache` path instead.

## Block-wise attention (step 3, rung 3)
- `paged_attention` is the payoff: `q @ K_block^T` per block, merged into a running softmax. `PagedKVCache.block()` returns a **view** of one block (a test asserts the data pointer matches), so attention multiplies straight against the pool. No tensor the length of the sequence is created anywhere.
- Why a running softmax: you cannot normalize before seeing every score, and keeping every score defeats the point. Keep three accumulators and rescale them when the max grows:
  `correction = exp(running_max - new_max)`, then `out = out * correction + exp(scores - new_max) @ V`. The first block's correction is 0 (the max starts at -inf), which is also why the initial state needs no special case.
- The result equals the one-shot softmax mathematically, not bit-wise: additions happen in block order. Measured 9e-8 in fp32, same argmax at every step, greedy tokens identical to HF. `torch.equal` still applies to *storage* (layer 0's K/V are a pure copy) but no longer to output logits.
- `PagedCache.update` no longer returns the history -- it hands back the step it was given. That API change is what makes the gather impossible rather than merely unused: attention must ask for blocks.
- `test_paged_attention_never_gathers_the_history` monkeypatches `gather` to raise. Without it a reintroduced gather is invisible, because the output would still be correct.
- Mask alignment: block b owns mask columns `[b * block_size, b * block_size + block_size)`. The last block is truncated to `take`, which also keeps unwritten slots out of the softmax -- a real `-inf` there would make a fully-masked query compute `exp(-inf - -inf)` = NaN.
- A fully masked query (left pad) still has *finite* scores (-3.4e38), so the running max is finite and the answer degrades to the uniform average, exactly like the contiguous path.
- Per-block costs that a kernel removes: `repeat_kv` duplicates KV heads (a kernel indexes heads), and `block()` stacks rows (a kernel reads per-row pointers). What is already gone is the big one: no sequence-length copy per layer per step.

## Questions to answer
- Prompt of 5 tokens: what is `logits.shape`, and what does each dimension mean?
- Why does the next token come only from the last position's logits?
- During cached decode, what shape is `input_ids`? What shape is K for one layer?
- In a left-padded batch, what is `logits[:, -1]` for the shortest prompt, and why is it not a pad?
- Position `p` of a row with 3 left pads: what does `cumsum(mask) - 1` give at its first real token, and why is 0 the right answer?
- Why does `position_ids` matter when feeding one token at a time?
- Why are `LLMEngine` and `ModelRunner` separate classes?
- `commit` runs after the last layer of a step. What breaks if it runs inside `write_layer`?
- During a step, does the causal mask need the committed length or committed + new? Why?
- In a batched paged cache, where does the mapping from (row, position) to memory live?
- Why does the running softmax need a max and a *weighted sum* of V, not just a sum of weights?
- What would go wrong if the last (partial) block were not truncated to `take`?
- Block b's scores come from `q @ K_block^T`. Which columns of the additive mask does that block own?
- KV cache in bf16 for 1000 tokens, in MB? How many tokens fit in a 3 GB budget in fp32 vs bf16?
