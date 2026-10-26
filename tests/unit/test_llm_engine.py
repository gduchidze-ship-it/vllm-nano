"""Engine loop tests with stubs: no model download, no tokenizer, no GPU.

These pin the stop rules and the argument validation, which are easy to break
while the engine is small.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_nano.engine.llm_engine import LLMEngine, eos_token_ids


class StubTokenizer:
    """Minimal tokenizer: a fixed prompt of token ids, and a decode that records flags."""

    def __init__(self, prompt_ids: tuple[int, ...] = (1, 2, 3)) -> None:
        self.prompt_ids = prompt_ids
        self.decode_kwargs: list[dict] = []

    def __call__(self, prompt: str, return_tensors: str | None = None) -> SimpleNamespace:
        return SimpleNamespace(input_ids=torch.tensor([list(self.prompt_ids)]))

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        self.decode_kwargs.append({"skip_special_tokens": skip_special_tokens})
        return "|".join(str(i) for i in ids)


class StubRunner:
    """Returns the scripted tokens in order and remembers what it was fed."""

    def __init__(self, tokens: list[int], vocab_size: int = 64) -> None:
        self.tokens = tokens
        self.vocab_size = vocab_size
        # (input_ids shape, did the call receive a cache?) per forward pass.
        self.calls: list[tuple[tuple[int, ...], bool]] = []

    def forward(
        self, input_ids: torch.Tensor, cache: object | None = None
    ) -> tuple[torch.Tensor, object]:
        self.calls.append((tuple(input_ids.shape), cache is not None))
        tok = self.tokens[len(self.calls) - 1]
        # One-hot-ish logits so argmax returns the scripted token.
        logits = torch.full((self.vocab_size,), -1.0)
        logits[tok] = 1.0
        return logits, object()  # the cache is opaque to the engine


def make_engine(
    tokens: list[int], eos: set[int] | None = None, max_model_len: int = 32
) -> tuple[LLMEngine, StubRunner]:
    runner = StubRunner(tokens)
    tokenizer = StubTokenizer()
    engine = LLMEngine(
        runner,  # type: ignore[arg-type]  # duck-typed stand-in for ModelRunner
        tokenizer,  # type: ignore[arg-type]  # duck-typed stand-in for PreTrainedTokenizerBase
        eos if eos is not None else set(),
        max_model_len,
    )
    return engine, runner


def test_stops_on_eos_and_keeps_the_token():
    engine, runner = make_engine([5, 9, 7], eos={9})

    result = engine.generate_ids("hi", max_new_tokens=10)

    assert result.token_ids == [5, 9]  # EOS is included, like HF
    assert result.finish_reason == "stop"
    assert len(runner.calls) == 2  # the loop stopped instead of using the budget


def test_stops_at_max_new_tokens():
    engine, runner = make_engine([5, 6, 7])

    result = engine.generate_ids("hi", max_new_tokens=2)

    assert result.token_ids == [5, 6]
    assert result.finish_reason == "length"
    assert len(runner.calls) == 2


def test_prefill_sees_the_prompt_then_one_token_per_step():
    engine, runner = make_engine([5, 6, 7])

    engine.generate_ids("hi", max_new_tokens=3)

    # One prefill of the 3-token prompt, then one token per step, always reusing the cache.
    assert runner.calls == [((1, 3), False), ((1, 1), True), ((1, 1), True)]


def test_rejects_non_positive_max_new_tokens():
    engine, runner = make_engine([5])

    with pytest.raises(ValueError, match="at least 1"):
        engine.generate_ids("hi", max_new_tokens=0)
    assert runner.calls == []  # validation happens before any forward pass


def test_rejects_prompt_plus_new_tokens_over_the_model_limit():
    engine, runner = make_engine([5], max_model_len=8)  # prompt 3 + 6 = 9 > 8

    with pytest.raises(ValueError, match="exceeds max_model_len"):
        engine.generate_ids("hi", max_new_tokens=6)
    assert runner.calls == []


def test_generate_decodes_without_special_tokens():
    engine, _ = make_engine([5])

    text = engine.generate("hi", max_new_tokens=1)

    assert text == "5"
    assert engine.tokenizer.decode_kwargs == [{"skip_special_tokens": True}]


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (None, set()),
        (7, {7}),
        ([7, 8], {7, 8}),
        ((7, 8), {7, 8}),
    ],
)
def test_eos_token_ids_accepts_every_generation_config_shape(configured, expected):
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=configured))

    assert eos_token_ids(model) == expected  # type: ignore[arg-type]
