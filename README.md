# vllm-nano

A minimal, hand-built LLM inference engine inspired by [vLLM](https://github.com/vllm-project/vllm).
Built **for learning**: every piece is written manually to understand how modern LLM serving works.

> Not production software. Readability over speed, understanding over features.

## Roadmap
- [ ] Single-request generation with a naive KV cache
- [ ] Batched generation with padding
- [ ] Paged KV cache (block allocator + block table)
- [ ] Scheduler + continuous batching
- [ ] Sampling strategies
- [ ] Simple API / CLI
- [ ] Stretch: prefix caching, CUDA graphs, tensor parallelism

## Verifying correctness
Greedy outputs should match Hugging Face `model.generate(...)` token for token.
Each stage of the roadmap should keep that test passing.

## References
- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [vLLM source](https://github.com/vllm-project/vllm)
- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)

## Notes
Learning notes and gotchas live in `NOTES.md` (worth keeping as you go).
