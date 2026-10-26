"""Timing: what batching bought us.

A decode step moves the same weights for every row, so extra requests in the batch
are nearly free. Compare batch=1 against batch=8: per-request cost should drop a
lot, while total tok/s should climb.

Prompts have different lengths on purpose: batching pads them, and the padding is
what step 3's paged cache will stop paying for.
"""

import statistics
import time

from loguru import logger

from vllm_nano.config import ModelConfig
from vllm_nano.engine.llm_engine import LLMEngine
from vllm_nano.engine.model_runner import ModelRunner
from vllm_nano.models.loader import load_model_and_tokenizer
from vllm_nano.utils.logging_setup import setup_logging

# Deliberately uneven lengths, so a batch has to pad.
PROMPTS = [
    "The capital of France is",
    "Write a long story about a dragon and a knight who",
    "2 + 2 =",
    "Explain in one sentence why the sky is blue:",
]
BATCH_SIZES = (1, 4, 8)
NEW_TOKENS = 50
REPEATS = 3


def main() -> None:
    setup_logging(log_dir=None)
    config = ModelConfig()
    model, tokenizer = load_model_and_tokenizer(config)
    # eos_ids=set(): never stop early, so every run produces exactly NEW_TOKENS per row.
    engine = LLMEngine(
        ModelRunner(model, config.resolved_device),
        tokenizer,
        eos_ids=set(),
        max_model_len=config.resolved_max_model_len,
    )

    engine.generate_batch([PROMPTS[0]], 5)  # warm-up: first call pays one-time GPU setup

    for batch_size in BATCH_SIZES:
        prompts = [PROMPTS[i % len(PROMPTS)] for i in range(batch_size)]
        times = []
        for _ in range(REPEATS):
            start = time.perf_counter()
            results = engine.generate_batch(prompts, NEW_TOKENS)
            times.append(time.perf_counter() - start)
            assert all(r.finish_reason == "length" for r in results)
            assert all(len(r.token_ids) == NEW_TOKENS for r in results)

        # Median ignores a slow outlier run (other apps, thermal throttling).
        median = statistics.median(times)
        total = batch_size * NEW_TOKENS
        logger.info(
            "batch {:>2}: {:.2f}s for {} tokens = {:.0f} tok/s total ({:.1f} tok/s per request)",
            batch_size,
            median,
            total,
            total / median,
            NEW_TOKENS / median,
        )


if __name__ == "__main__":
    main()
