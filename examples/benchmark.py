"""Timing: what the KV cache bought us.

With a cache, a decode step does the same work no matter how long the sequence is,
so tok/s should stay roughly FLAT as the token count grows. Before the cache, every
step re-ran the whole sequence: 100 tokens was more than 2x slower per token than 25.
Run this on the commit before the cache to see that difference.
"""

import statistics
import time

from loguru import logger

from vllm_nano.config import ModelConfig
from vllm_nano.engine.llm_engine import LLMEngine
from vllm_nano.engine.model_runner import ModelRunner
from vllm_nano.models.loader import load_model_and_tokenizer
from vllm_nano.utils.logging_setup import setup_logging

PROMPT = "Write a long story about a dragon and a knight who"
TOKEN_COUNTS = (25, 100)
REPEATS = 3


def main() -> None:
    setup_logging(log_dir=None)
    config = ModelConfig()
    model, tokenizer = load_model_and_tokenizer(config)
    # eos_ids=set(): never stop early, so every run produces exactly N tokens.
    engine = LLMEngine(
        ModelRunner(model, config.resolved_device),
        tokenizer,
        eos_ids=set(),
        max_model_len=config.resolved_max_model_len,
    )

    engine.generate_ids(PROMPT, 5)  # warm-up: the first call pays one-time GPU setup

    for new_tokens in TOKEN_COUNTS:
        times = []
        for _ in range(REPEATS):
            start = time.perf_counter()
            result = engine.generate_ids(PROMPT, new_tokens)
            times.append(time.perf_counter() - start)
            assert result.finish_reason == "length"
            assert len(result.token_ids) == new_tokens

        # Median ignores a slow outlier run (other apps, thermal throttling).
        median = statistics.median(times)
        logger.info(
            "{:>3} tokens: {:.2f}s = {:.1f} tok/s ({:.1f} ms/token)",
            new_tokens,
            median,
            new_tokens / median,
            1000 * median / new_tokens,
        )


if __name__ == "__main__":
    main()
