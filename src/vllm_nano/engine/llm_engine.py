from transformers import PreTrainedModel, PreTrainedTokenizerBase

from vllm_nano.engine.model_runner import ModelRunner
from vllm_nano.sequence import GenerationResult


def eos_token_ids(model: PreTrainedModel) -> set[int]:
    """End-of-sequence ids from the model's generation config.

    An empty set means the model defines no EOS, so only `max_new_tokens` stops it.
    """
    eos = model.generation_config.eos_token_id
    if eos is None:
        return set()
    if isinstance(eos, (list, tuple)):
        return set(eos)
    return {eos}


class LLMEngine:
    """Owns the generation loop and the stop rules; the runner owns the model."""

    def __init__(
        self,
        runner: ModelRunner,
        tokenizer: PreTrainedTokenizerBase,
        eos_ids: set[int],
        max_model_len: int,
    ) -> None:
        self.runner = runner
        self.tokenizer = tokenizer
        self.eos_ids = eos_ids
        self.max_model_len = max_model_len

    def generate_ids(self, prompt: str, max_new_tokens: int) -> GenerationResult:
        """Greedy-decode up to `max_new_tokens`; the EOS token is included, like HF."""
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be at least 1, got {max_new_tokens}")

        ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        prompt_len = ids.shape[1]  # (batch=1, seq_len); no padding yet, so this is real
        if prompt_len + max_new_tokens > self.max_model_len:
            raise ValueError(
                f"prompt ({prompt_len}) + max_new_tokens ({max_new_tokens}) exceeds "
                f"max_model_len ({self.max_model_len})"
            )

        # Prefill: one pass over the prompt fills the cache. Every step after this
        # feeds a single token, so the work per token is constant instead of O(n).
        logits, cache = self.runner.forward(ids)
        new_tokens: list[int] = []
        finish_reason: str = "length"
        for step in range(max_new_tokens):
            tok = int(logits.argmax())  # greedy; a sampler replaces this at step 5
            new_tokens.append(tok)  # append BEFORE the check so EOS is kept
            if tok in self.eos_ids:
                finish_reason = "stop"
                break
            if step + 1 < max_new_tokens:  # no point decoding a token nobody will read
                # .new_tensor keeps the dtype and device of `ids`.
                logits, cache = self.runner.forward(ids.new_tensor([[tok]]), cache)
        return GenerationResult(token_ids=new_tokens, finish_reason=finish_reason)

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        """Text version of `generate_ids`, for humans."""
        result = self.generate_ids(prompt, max_new_tokens)
        text = self.tokenizer.decode(result.token_ids, skip_special_tokens=True)
        if not isinstance(text, str):
            raise TypeError(f"Expected str, received {type(text)}")
        return text
