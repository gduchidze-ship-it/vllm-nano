import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from vllm_nano.engine.model_runner import ModelRunner


def _eos_ids(model: PreTrainedModel) -> set[int]:
    eos = model.generation_config.eos_token_id
    if eos is None:
        return set()  # model defines no EOS: only max_new_tokens stops generation
    return set(eos) if isinstance(eos, list) else {eos}  # {eos}: a set with one item

class LLMEngine:
    """Owns the generation loop and the stop rules; the runner owns the model."""

    def __init__(
        self,
        runner: ModelRunner,
        tokenizer: PreTrainedTokenizerBase,
        eos_ids: set[int],
    ) -> None:
        self.runner = runner
        self.tokenizer = tokenizer
        self.eos_ids = eos_ids

    def generate_ids(self, prompt: str, max_new_tokens: int) -> list[int]:
        """Greedy-decode up to `max_new_tokens`; the EOS token is included, like HF."""
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        new_tokens: list[int] = []
        for _ in range(max_new_tokens):
            tok = self.runner.next_token(ids)  # forward pass over the whole sequence
            new_tokens.append(tok)  # append BEFORE the check so EOS is kept
            if tok in self.eos_ids:
                break
            ids = torch.cat([ids, torch.tensor([[tok]])], dim=1)  # (1, seq_len + 1)
        return new_tokens

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        """Text version of `generate_ids`, for humans."""
        text = self.tokenizer.decode(self.generate_ids(prompt, max_new_tokens))
        if not isinstance(text, str):
            raise TypeError(f"Expected str, received {type(text)}")
        return text
