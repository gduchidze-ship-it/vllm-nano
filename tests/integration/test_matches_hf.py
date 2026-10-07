import pytest

from vllm_nano.config import ModelConfig
from vllm_nano.engine.llm_engine import LLMEngine, _eos_ids
from vllm_nano.engine.model_runner import ModelRunner
from vllm_nano.models.loader import load_model_and_tokenizer

PROMPTS = [
    "The capital of France is",
    "Write a long story about a dragon and a knight who",
    "2 + 2 =",  # likely finishes early: exercises the EOS path
]
MAX_NEW_TOKENS = 30


@pytest.fixture(scope="module")
def stack():
    """Load the model once for the whole file (loading takes seconds)."""
    config = ModelConfig()
    model, tokenizer = load_model_and_tokenizer(config)
    engine = LLMEngine(ModelRunner(model, config.resolved_device), tokenizer, _eos_ids(model))
    return engine, model, tokenizer, config


def _first_diff(a: list[int], b: list[int]) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


@pytest.mark.parametrize("prompt", PROMPTS)
def test_greedy_matches_hf(stack, prompt):
    engine, model, tokenizer, config = stack
    inputs = tokenizer(prompt, return_tensors="pt").to(config.resolved_device)
    # repetition_penalty=1.0: Qwen's generation_config sets 1.1, which HF applies even greedy.
    out = model.generate(
        **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, repetition_penalty=1.0
    )
    expected = out[0, inputs.input_ids.shape[1]:].tolist()  # drop the prompt

    actual = engine.generate_ids(prompt, MAX_NEW_TOKENS)

    assert actual == expected, f"first mismatch at step {_first_diff(actual, expected)}"
