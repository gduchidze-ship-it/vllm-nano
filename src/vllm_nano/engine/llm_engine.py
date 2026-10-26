import torch
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
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def generate_ids(self, prompt: str, max_new_tokens: int) -> GenerationResult:
        """One prompt: a batch of one. Kept because most tests and examples want this."""
        return self.generate_batch([prompt], max_new_tokens)[0]

    def generate_batch(self, prompts: list[str], max_new_tokens: int) -> list[GenerationResult]:
        """Greedy-decode a batch; one forward pass per step covers every row.

        All sequences must be rectangular, so prompts are padded. LEFT padding keeps
        every real sequence flush against the right edge, which is what makes
        `logits[:, -1]` the next-token position for every row.
        """
        if not prompts:
            raise ValueError("prompts must not be empty")
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be at least 1, got {max_new_tokens}")

        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
        ids = encoded.input_ids  # (batch, longest prompt); shorter prompts start with pads
        mask = encoded.attention_mask  # (batch, longest prompt); 0 = pad
        batch, prompt_len = ids.shape
        if prompt_len + max_new_tokens > self.max_model_len:
            raise ValueError(
                f"longest prompt ({prompt_len}) + max_new_tokens ({max_new_tokens}) "
                f"exceeds max_model_len ({self.max_model_len})"
            )

        # Left pads must not shift a sequence's positions. Rotate-embedding models do
        # not care about a uniform shift (positions are relative), but absolute-position
        # and ALiBi models do, and this is the model-agnostic way to be right.
        position_ids = (mask.cumsum(-1) - 1).clamp(min=0)

        logits, cache = self.runner.forward(ids, attention_mask=mask, position_ids=position_ids)

        tokens: list[list[int]] = [[] for _ in range(batch)]
        finished = [False] * batch
        reasons: list[str] = ["length"] * batch  # until proven otherwise

        for step in range(max_new_tokens):
            next_ids = logits.argmax(-1)  # (batch,): greedy, a sampler replaces this at step 5
            for row in range(batch):
                if finished[row]:
                    continue  # already reported; its slot is still occupied until the batch ends
                tok = int(next_ids[row])
                tokens[row].append(tok)  # append BEFORE the check so EOS is kept
                if tok in self.eos_ids:
                    finished[row] = True
                    reasons[row] = "stop"
                elif step + 1 == max_new_tokens:
                    finished[row] = True

            if step + 1 == max_new_tokens or all(finished):
                break

            feed = next_ids.clone()
            feed[finished] = self.pad_token_id
            mask = torch.cat([mask, mask.new_ones((batch, 1))], dim=1)
            position_ids = (mask.cumsum(-1) - 1)[:, -1:]
            logits, cache = self.runner.forward(
                feed.view(batch, 1),
                attention_mask=mask,
                position_ids=position_ids,
                cache=cache,
            )

        return [GenerationResult(tokens[row], reasons[row]) for row in range(batch)]

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        """Text version of `generate_ids`, for humans."""
        result = self.generate_ids(prompt, max_new_tokens)
        text = self.tokenizer.decode(result.token_ids, skip_special_tokens=True)
        if not isinstance(text, str):
            raise TypeError(f"Expected str, received {type(text)}")
        return text
