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

## Questions to answer
- Prompt of 5 tokens: what is `logits.shape`, and what does each dimension mean?
- Why does the next token come only from the last position's logits?
- During cached decode, what shape is `input_ids`? What shape is K for one layer?
- Why does `position_ids` matter when feeding one token at a time?
- Why are `LLMEngine` and `ModelRunner` separate classes?
- KV cache in bf16 for 1000 tokens, in MB? How many tokens fit in a 3 GB budget in fp32 vs bf16?
