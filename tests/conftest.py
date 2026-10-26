"""Shared fixtures.

The tiny model is a randomly-initialized Qwen2 built from a config dict: the same
architecture family as Qwen2.5-0.5B (GQA, RoPE, the same forward signature) but 2
layers and 32 hidden units, so tests run on CPU in milliseconds and need no download.
"""

import pytest
import torch
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

# hidden_size = num_attention_heads * head_dim = 4 * 8, so head_dim resolves to 8.
TINY_CONFIG = {
    "vocab_size": 128,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,  # GQA: 4 query heads share 2 KV heads
    "max_position_embeddings": 64,
}

TINY_VOCAB_SIZE = TINY_CONFIG["vocab_size"]


@pytest.fixture
def tiny_model() -> Qwen2ForCausalLM:
    torch.manual_seed(0)  # same weights every run, so exact assertions stay stable
    return Qwen2ForCausalLM(Qwen2Config(**TINY_CONFIG)).eval()
