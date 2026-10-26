"""ModelRunner tests against a real (tiny) model on CPU, no download.

The cached-decode and padded-batch tests are the ones that matter: they pin the
invariants the KV-cache and batching designs rest on, and they run in milliseconds.
"""

import torch

from vllm_nano.engine.model_runner import ModelRunner

PROMPT = torch.tensor([[1, 2, 3, 4, 5]])
LAST = torch.tensor([[7]])
SEQUENCE = torch.tensor([[1, 2, 3, 4, 5, 7]])


def test_forward_returns_one_logit_row_per_sequence(tiny_model):
    logits, _ = ModelRunner(tiny_model, "cpu").forward(PROMPT)

    assert logits.shape == (1, tiny_model.config.vocab_size)


def test_forward_is_deterministic(tiny_model):
    # Greedy decoding must be reproducible; a stray dropout or state leak breaks this.
    runner = ModelRunner(tiny_model, "cpu")

    first, _ = runner.forward(PROMPT)
    second, _ = runner.forward(PROMPT)

    assert torch.equal(first, second)


def test_forward_matches_hf_greedy_generate(tiny_model):
    """Cross-check against HF's generate(), not against our own argmax again."""
    expected = tiny_model.generate(PROMPT, max_new_tokens=1, do_sample=False)[0, -1].item()

    logits, _ = ModelRunner(tiny_model, "cpu").forward(PROMPT)

    assert int(logits[0].argmax()) == expected


def test_cached_decode_matches_one_full_forward(tiny_model):
    """Prefill + one decode must equal a single forward pass over the whole sequence.

    This is the invariant: logits at position t depend only on tokens <= t, so the
    cache is just a shortcut, not an approximation.
    """
    runner = ModelRunner(tiny_model, "cpu")
    full = tiny_model(SEQUENCE, use_cache=False).logits[0, -1]

    _, cache = runner.forward(PROMPT)
    logits, _ = runner.forward(LAST, cache=cache)

    assert torch.allclose(logits[0], full, atol=1e-6)
    assert int(logits[0].argmax()) == int(full.argmax())


def test_cache_holds_one_entry_per_token(tiny_model):
    runner = ModelRunner(tiny_model, "cpu")

    _, cache = runner.forward(PROMPT)
    assert cache.get_seq_length() == PROMPT.shape[1]

    _, cache = runner.forward(LAST, cache=cache)
    assert cache.get_seq_length() == SEQUENCE.shape[1]


def test_cache_bytes_per_token_matches_the_formula(tiny_model):
    """2 (K and V) x layers x kv_heads x head_dim x bytes, exactly as NOTES.md says."""
    cfg = tiny_model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    expected = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * 4

    _, cache = ModelRunner(tiny_model, "cpu").forward(PROMPT)
    measured = (
        sum(
            layer.keys.numel() * layer.keys.element_size()
            + layer.values.numel() * layer.values.element_size()
            for layer in cache.layers
        )
        / PROMPT.shape[1]
    )

    assert measured == expected


def test_passing_the_whole_sequence_with_a_warm_cache_is_wrong(tiny_model):
    """The classic bug: the cache already holds the prompt, so sending it again
    makes the model attend to every prompt token twice. Assert we notice."""
    runner = ModelRunner(tiny_model, "cpu")
    full = tiny_model(SEQUENCE, use_cache=False).logits[0, -1]

    _, cache = runner.forward(PROMPT)
    wrong, _ = runner.forward(SEQUENCE, cache=cache)

    assert not torch.allclose(wrong[0], full)


# A left-padded batch: row 0 is "6 7" behind three pads, row 1 is the 5-token prompt.
PADDED_IDS = torch.tensor([[0, 0, 0, 6, 7], [1, 2, 3, 4, 5]])
PADDED_MASK = torch.tensor([[0, 0, 0, 1, 1], [1, 1, 1, 1, 1]])
PADDED_POSITIONS = (PADDED_MASK.cumsum(-1) - 1).clamp(min=0)


def test_left_padded_batch_matches_the_same_rows_run_alone(tiny_model):
    """Batching must not change results: padding is bookkeeping, not context."""
    runner = ModelRunner(tiny_model, "cpu")

    batched, _ = runner.forward(
        PADDED_IDS, attention_mask=PADDED_MASK, position_ids=PADDED_POSITIONS
    )
    short_alone, _ = runner.forward(torch.tensor([[6, 7]]))
    long_alone, _ = runner.forward(PROMPT)

    assert torch.allclose(batched[0], short_alone[0], atol=1e-5)
    assert torch.allclose(batched[1], long_alone[0], atol=1e-5)


def test_mask_must_cover_the_cache_on_every_decode_step(tiny_model):
    """Forgetting to extend the mask makes the model attend to the wrong tokens.

    The stale mask is accepted silently, so only a test like this catches it.
    """
    runner = ModelRunner(tiny_model, "cpu")
    full_mask = torch.ones(1, SEQUENCE.shape[1], dtype=torch.long)
    positions = torch.tensor([[SEQUENCE.shape[1] - 1]])

    def fresh_cache():
        return runner.forward(PROMPT)[1]

    correct, _ = runner.forward(
        LAST, attention_mask=full_mask, position_ids=positions, cache=fresh_cache()
    )
    # 5-wide mask with a 5-token cache: the new token is not covered.
    stale, _ = runner.forward(
        LAST, attention_mask=torch.ones(1, PROMPT.shape[1]), cache=fresh_cache()
    )

    assert not torch.allclose(correct[0], stale[0])
