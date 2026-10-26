"""ModelRunner tests against a real (tiny) model on CPU, no download.

The cached-decode test is the one that matters: it pins the invariant the whole
KV-cache design rests on, and it runs in milliseconds.
"""

import torch

from vllm_nano.engine.model_runner import ModelRunner

# Same numbers as the tiny model in conftest.py: a 6-token sequence, plus the prefix
# that prefills the cache before the last token is decoded one at a time.
PROMPT = torch.tensor([[1, 2, 3, 4, 5]])
LAST = torch.tensor([[7]])
SEQUENCE = torch.tensor([[1, 2, 3, 4, 5, 7]])


def test_forward_returns_one_logit_per_vocab_entry(tiny_model):
    logits, _ = ModelRunner(tiny_model, "cpu").forward(PROMPT)

    assert logits.shape == (tiny_model.config.vocab_size,)


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

    assert int(logits.argmax()) == expected


def test_cached_decode_matches_one_full_forward(tiny_model):
    """Prefill + one decode must equal a single forward pass over the whole sequence.

    This is the invariant: logits at position t depend only on tokens <= t, so the
    cache is just a shortcut, not an approximation.
    """
    runner = ModelRunner(tiny_model, "cpu")
    full = tiny_model(SEQUENCE, use_cache=False).logits[0, -1]

    _, cache = runner.forward(PROMPT)
    logits, _ = runner.forward(LAST, cache)

    assert torch.allclose(logits, full, atol=1e-6)
    assert int(logits.argmax()) == int(full.argmax())


def test_cache_holds_one_entry_per_token(tiny_model):
    runner = ModelRunner(tiny_model, "cpu")

    _, cache = runner.forward(PROMPT)
    assert cache.get_seq_length() == PROMPT.shape[1]

    _, cache = runner.forward(LAST, cache)
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
    wrong, _ = runner.forward(SEQUENCE, cache)

    assert not torch.allclose(wrong, full)
