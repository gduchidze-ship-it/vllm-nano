import pytest

from vllm_nano.config import ModelConfig
from vllm_nano.engine.llm_engine import LLMEngine, eos_token_ids
from vllm_nano.engine.model_runner import ModelRunner
from vllm_nano.models.loader import load_model_and_tokenizer

PROMPTS = [
    "The capital of France is",
    "Write a long story about a dragon and a knight who",
    "2 + 2 =",  # likely finishes early: exercises the EOS path
]
MAX_NEW_TOKENS = 30

# Downloads ~1 GB and needs a few seconds per prompt: opt in with `pytest -m slow`.
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def stack():
    """Load the model once for the whole file (loading takes seconds)."""
    config = ModelConfig()
    model, tokenizer = load_model_and_tokenizer(config)
    engine = LLMEngine(
        ModelRunner(model, config.resolved_device),
        tokenizer,
        eos_token_ids(model),
        config.resolved_max_model_len,
    )
    return engine, model, tokenizer, config


def _first_diff(a: list[int], b: list[int]) -> int:
    return next(
        (i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b))
    )


@pytest.mark.parametrize("prompt", PROMPTS)
def test_greedy_matches_hf(stack, prompt):
    engine, model, tokenizer, config = stack
    inputs = tokenizer(prompt, return_tensors="pt").to(config.resolved_device)
    # repetition_penalty=1.0: Qwen's generation_config sets 1.1, which HF applies even greedy.
    out = model.generate(
        **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, repetition_penalty=1.0
    )
    expected = out[0, inputs.input_ids.shape[1] :].tolist()  # drop the prompt

    actual = engine.generate_ids(prompt, MAX_NEW_TOKENS)

    diff_at = _first_diff(actual.token_ids, expected)
    assert actual.token_ids == expected, f"first mismatch at step {diff_at}"
    # Same story for the stop reason: EOS in the reference output means we stopped on it.
    expected_reason = "stop" if expected and expected[-1] in eos_token_ids(model) else "length"
    assert actual.finish_reason == expected_reason


BATCH_PROMPTS = ["2 + 2 =", "The capital of France is", "Write a short poem about the sea"]


def test_batched_greedy_matches_hf(stack):
    """One forward per step for 3 rows of different lengths must still match HF.

    This is the real test of left padding + position_ids: a wrong pad side or a
    position shift shows up as a token mismatch here.
    """
    engine, model, tokenizer, config = stack
    results = engine.generate_batch(BATCH_PROMPTS, MAX_NEW_TOKENS)

    for prompt, result in zip(BATCH_PROMPTS, results, strict=True):
        inputs = tokenizer(prompt, return_tensors="pt").to(config.resolved_device)
        out = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, repetition_penalty=1.0
        )
        expected = out[0, inputs.input_ids.shape[1] :].tolist()
        diff_at = _first_diff(result.token_ids, expected)
        assert result.token_ids == expected, f"batch mismatch for {prompt!r} at step {diff_at}"
