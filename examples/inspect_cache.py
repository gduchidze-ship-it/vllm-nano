"""Look inside HF's KV cache: its type, shapes, size, and how it grows.

Two phases:
    PREFILL: the whole prompt goes in at once; the cache is filled.
    DECODE:  one new token goes in, plus the cache; the cache grows by one.
"""

import torch
from loguru import logger

from vllm_nano.config import ModelConfig
from vllm_nano.models.loader import load_model_and_tokenizer
from vllm_nano.utils.logging_setup import setup_logging


def main() -> None:
    setup_logging(log_dir=None)
    config = ModelConfig()
    model, tokenizer = load_model_and_tokenizer(config)
    inputs = tokenizer("The capital of France is", return_tensors="pt").to(config.resolved_device)
    prompt_len = inputs.input_ids.shape[1]

    with torch.inference_mode():
        # PREFILL: run the whole prompt; use_cache=True asks the model to return its cache.
        out = model(**inputs, use_cache=True)
        cache = out.past_key_values
        layer0 = cache.layers[0]
        logger.info(
            "cache type {}, layers {}, seq_len {}",
            type(cache).__name__, len(cache), cache.get_seq_length(),
        )
        # Shape: (batch, kv_heads, seq_len, head_dim) = (1, 2, 5, 64) for Qwen2.5-0.5B.
        logger.info("layer 0 K {} V {}", tuple(layer0.keys.shape), tuple(layer0.values.shape))

        # Check the memory formula from NOTES.md against the real tensors:
        # bytes per token = 2 (K and V) x layers x kv_heads x head_dim x bytes per number.
        total = sum(
            lyr.keys.numel() * lyr.keys.element_size()
            + lyr.values.numel() * lyr.values.element_size()
            for lyr in cache.layers
        )
        formula = (
            2 * config.num_layers * config.num_kv_heads * config.head_dim
            * torch.finfo(config.torch_dtype).bits // 8
        )
        logger.info("bytes per token: measured {}, formula {}", total / prompt_len, formula)

        # DECODE: only the newest token goes in; the model reads the old K/V from the cache.
        # The new token's position (5) comes from the cache length, not from us.
        next_id = out.logits[0, -1].argmax().view(1, 1)  # shape (1, 1)
        out2 = model(input_ids=next_id, past_key_values=cache, use_cache=True)
        logger.info(
            "after decode: seq_len {}, layer 0 K {}",
            cache.get_seq_length(), tuple(cache.layers[0].keys.shape),
        )
        # The model appended to the cache we passed in (in place) and returned the same object.
        logger.info("same cache object returned: {}", out2.past_key_values is cache)


if __name__ == "__main__":
    main()
