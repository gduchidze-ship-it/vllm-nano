"""Our hand-written attention must reproduce HF's numbers before we change anything.

If these pass, the copy is faithful. The paged version (next step) will be judged
against *this* implementation, and against the same token-level oracle.
"""

import copy

import torch

from vllm_nano.models.attention import swap_attention

IDS = torch.tensor([[1, 2, 3, 4, 5]])
ATOL = 1e-6  # fp32 rounding between fused SDPA and our explicit matmuls


def swapped(model):
    """A copy with our attention in it, so the original stays usable as the oracle."""
    clone = copy.deepcopy(model)
    swap_attention(clone)
    return clone


def test_swapped_model_matches_hf_logits_on_prefill(tiny_model):
    ours = swapped(tiny_model)

    with torch.inference_mode():
        expected = tiny_model(IDS).logits
        actual = ours(IDS).logits

    assert torch.allclose(actual, expected, atol=ATOL)
    assert torch.equal(actual.argmax(-1), expected.argmax(-1))


def test_swapped_model_writes_the_same_cache(tiny_model):
    """K/V are the numbers the paged cache will store, so they must be identical."""
    ours = swapped(tiny_model)

    with torch.inference_mode():
        expected = tiny_model(IDS, use_cache=True).past_key_values
        actual = ours(IDS, use_cache=True).past_key_values

    for layer in range(tiny_model.config.num_hidden_layers):
        assert torch.allclose(actual.layers[layer].keys, expected.layers[layer].keys, atol=ATOL)
        assert torch.allclose(actual.layers[layer].values, expected.layers[layer].values, atol=ATOL)


def test_swapped_model_matches_hf_for_several_decode_steps(tiny_model):
    """Prefill is one call; the interesting bugs show up when the cache is read back."""
    ours = swapped(tiny_model)

    with torch.inference_mode():
        expected = tiny_model(IDS, use_cache=True)
        actual = ours(IDS, use_cache=True)
        for step in range(4):
            token = expected.logits[:, -1].argmax(-1, keepdim=True)
            expected = tiny_model(token, past_key_values=expected.past_key_values, use_cache=True)
            actual = ours(token, past_key_values=actual.past_key_values, use_cache=True)
            assert torch.allclose(actual.logits, expected.logits, atol=ATOL), f"step {step}"


def test_swapped_model_generates_the_same_tokens(tiny_model):
    """End to end: HF's generate() runs our attention and must pick the same tokens."""
    ours = swapped(tiny_model)

    expected = tiny_model.generate(IDS, max_new_tokens=5, do_sample=False)
    actual = ours.generate(IDS, max_new_tokens=5, do_sample=False)

    assert torch.equal(actual, expected)


def test_swap_declares_itself_eager(tiny_model):
    """Without this flag the model passes no mask and attention becomes bidirectional.

    That is a silent correctness bug: plausible text, wrong tokens. Pinned here.
    """
    ours = swapped(tiny_model)

    assert ours.config._attn_implementation == "eager"


def test_attention_is_causal_even_after_the_swap(tiny_model):
    """Position 0 sees only token 0, so changing later tokens must not move its logits.

    This fails loudly if the causal mask ever goes missing.
    """
    ours = swapped(tiny_model)
    changed = torch.tensor([[1, 9, 8, 7, 6]])  # same first token, different continuation

    with torch.inference_mode():
        first = ours(IDS).logits[0, 0]
        second = ours(changed).logits[0, 0]

    assert torch.allclose(first, second, atol=ATOL)
