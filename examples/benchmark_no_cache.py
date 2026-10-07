"""Baseline timing: generation WITHOUT a KV cache.

Every step re-runs the whole sequence through the model, so time per token
grows as the text gets longer. This is the number the KV cache must beat.
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
NEW_TOKENS = 50
REPEATS = 3


def main() -> None:
    setup_logging(log_dir=None)
    config = ModelConfig()
    model, tokenizer = load_model_and_tokenizer(config)
    # eos_ids=set(): never stop early, so every run produces exactly NEW_TOKENS.
    engine = LLMEngine(ModelRunner(model, config.resolved_device), tokenizer, eos_ids=set())

    engine.generate_ids(PROMPT, 5)  # warm-up: the first call pays one-time GPU setup

    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        ids = engine.generate_ids(PROMPT, NEW_TOKENS)
        times.append(time.perf_counter() - start)
        assert len(ids) == NEW_TOKENS

    # Median ignores a slow outlier run (other apps, thermal throttling).
    median = statistics.median(times)
    logger.info(
        "no cache: {:.2f}s for {} tokens = {:.1f} tok/s", median, NEW_TOKENS, NEW_TOKENS / median
    )


if __name__ == "__main__":
    main()
