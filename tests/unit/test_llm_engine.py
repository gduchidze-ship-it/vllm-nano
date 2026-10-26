"""Engine loop and batching tests with stubs: no model download, no tokenizer, no GPU.

These pin the stop rules, the left-padding contract, and the batch bookkeeping that
are easy to break while the engine is small.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_nano.engine.llm_engine import LLMEngine, eos_token_ids


class StubTokenizer:
    """Tokenizer stand-in: scripted prompt ids, and real left-padding behaviour."""

    def __init__(self, prompt_ids: list[tuple[int, ...]] | None = None) -> None:
        self.prompt_ids = prompt_ids if prompt_ids is not None else [(1, 2, 3)]
        self.pad_token_id = 0
        self.decode_kwargs: list[dict] = []

    def __call__(
        self,
        prompts: str | list[str],
        return_tensors: str | None = None,
        padding: bool = False,
        padding_side: str = "right",
    ) -> SimpleNamespace:
        if isinstance(prompts, str):
            prompts = [prompts]
        rows = [list(ids) for ids in self.prompt_ids[: len(prompts)]]
        width = max(len(row) for row in rows) if padding else len(rows[0])
        ids_out, mask_out = [], []
        for row in rows:
            pads = [self.pad_token_id] * (width - len(row))
            if padding_side == "left":
                ids_out.append(pads + row)
                mask_out.append([0] * len(pads) + [1] * len(row))
            else:
                ids_out.append(row + pads)
                mask_out.append([1] * len(row) + [0] * len(pads))
        return SimpleNamespace(
            input_ids=torch.tensor(ids_out), attention_mask=torch.tensor(mask_out)
        )

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        self.decode_kwargs.append({"skip_special_tokens": skip_special_tokens})
        return "|".join(str(i) for i in ids)


class StubRunner:
    """Scripted logits per (row, step), so the engine's bookkeeping is observable."""

    def __init__(self, scripts: list[list[int]], vocab_size: int = 64) -> None:
        self.scripts = scripts
        self.vocab_size = vocab_size
        self.calls: list[dict] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        cache: object | None = None,
    ) -> tuple[torch.Tensor, object]:
        step = len(self.calls)
        self.calls.append(
            {
                "shape": tuple(input_ids.shape),
                "has_cache": cache is not None,
                "mask": attention_mask.tolist() if attention_mask is not None else None,
                "positions": position_ids.tolist() if position_ids is not None else None,
            }
        )
        batch = input_ids.shape[0]
        logits = torch.full((batch, self.vocab_size), -1.0)
        for row in range(batch):
            logits[row, self.scripts[row][step]] = 1.0  # argmax returns the scripted token
        return logits, object()  # the cache is opaque to the engine


def make_engine(
    tokens: list[int],
    eos: set[int] | None = None,
    max_model_len: int = 32,
    prompt_ids: list[tuple[int, ...]] | None = None,
) -> tuple[LLMEngine, StubRunner]:
    runner = StubRunner([tokens])
    engine = LLMEngine(
        runner,  # type: ignore[arg-type]  # duck-typed stand-in for ModelRunner
        StubTokenizer(prompt_ids),  # type: ignore[arg-type]  # duck-typed tokenizer
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
    assert [call["shape"] for call in runner.calls] == [(1, 3), (1, 1), (1, 1)]
    assert [call["has_cache"] for call in runner.calls] == [False, True, True]


def test_one_forward_per_step_covers_the_whole_batch():
    runner = StubRunner([[5, 6, 7], [8, 9, 10]])
    engine = LLMEngine(
        runner,  # type: ignore[arg-type]
        StubTokenizer([(1, 2), (3, 4, 5)]),  # type: ignore[arg-type]
        eos_ids=set(),
        max_model_len=32,
    )

    results = engine.generate_batch(["short", "long"], max_new_tokens=3)

    assert [r.token_ids for r in results] == [[5, 6, 7], [8, 9, 10]]
    # Both rows ride in the same three forward passes, not six.
    assert [call["shape"] for call in runner.calls] == [(2, 3), (2, 1), (2, 1)]
    assert [call["has_cache"] for call in runner.calls] == [False, True, True]


def test_prefill_left_pads_and_positions_ignore_the_pads():
    runner = StubRunner([[5, 6], [7, 8]])
    engine = LLMEngine(
        runner,  # type: ignore[arg-type]
        StubTokenizer([(1, 2), (3, 4, 5)]),  # type: ignore[arg-type]
        eos_ids=set(),
        max_model_len=32,
    )

    engine.generate_batch(["short", "long"], max_new_tokens=2)

    # Shorter prompt gets one pad on the LEFT; its first real token is position 0.
    assert runner.calls[0]["mask"] == [[0, 1, 1], [1, 1, 1]]
    assert runner.calls[0]["positions"] == [[0, 0, 1], [0, 1, 2]]


def test_decode_extends_the_mask_and_advances_positions():
    runner = StubRunner([[5, 6], [7, 8]])
    engine = LLMEngine(
        runner,  # type: ignore[arg-type]
        StubTokenizer([(1, 2), (3, 4, 5)]),  # type: ignore[arg-type]
        eos_ids=set(),
        max_model_len=32,
    )

    engine.generate_batch(["short", "long"], max_new_tokens=2)

    # The mask covers cache + new token, so 3 real tokens -> width 4 on the first decode.
    assert runner.calls[1]["mask"] == [[0, 1, 1, 1], [1, 1, 1, 1]]
    # Positions are per-row: the short prompt's next token is its 3rd (index 2).
    assert runner.calls[1]["positions"] == [[2], [3]]


def test_a_finished_row_does_not_stop_or_alter_the_others():
    # Row 0 hits EOS on its first token; row 1 keeps going to the budget.
    runner = StubRunner([[9, 0, 0], [5, 6, 7]])
    engine = LLMEngine(
        runner,  # type: ignore[arg-type]
        StubTokenizer([(1, 2), (3, 4)]),  # type: ignore[arg-type]
        eos_ids={9},
        max_model_len=32,
    )

    results = engine.generate_batch(["a", "b"], max_new_tokens=3)

    assert results[0].token_ids == [9]
    assert results[0].finish_reason == "stop"
    assert results[1].token_ids == [5, 6, 7]
    assert results[1].finish_reason == "length"


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


def test_limit_is_checked_against_the_longest_prompt():
    runner = StubRunner([[5], [6]])
    engine = LLMEngine(
        runner,  # type: ignore[arg-type]
        StubTokenizer([(1, 2), (3, 4, 5)]),  # type: ignore[arg-type]
        eos_ids=set(),
        max_model_len=8,  # longest prompt 3 + 6 = 9 > 8, even though the other is short
    )

    with pytest.raises(ValueError, match="longest prompt"):
        engine.generate_batch(["short", "long"], max_new_tokens=6)
    assert runner.calls == []


def test_rejects_empty_prompt_list():
    engine, _ = make_engine([5])

    with pytest.raises(ValueError, match="must not be empty"):
        engine.generate_batch([], max_new_tokens=1)


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
