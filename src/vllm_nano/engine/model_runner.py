import torch
from transformers import PreTrainedModel
from transformers.cache_utils import Cache


class ModelRunner:
    """Owns the model and runs forward passes; knows nothing about stopping or scheduling."""

    def __init__(self, model: PreTrainedModel, device: str) -> None:
        self.model = model
        self.device = device

    @torch.inference_mode()  # no gradients on every call, so it cannot be forgotten
    def forward(
        self, input_ids: torch.Tensor, cache: Cache | None = None
    ) -> tuple[torch.Tensor, Cache]:
        """Run one forward pass; return the LAST position's logits and the KV cache.

        `input_ids` is (1, L). The first call passes the whole prompt (prefill) and
        returns a cache holding L tokens. Every later call passes exactly one token
        (decode): the model reads the past K/V from `cache` and appends its own, so
        the cost of a step no longer grows with the sequence.
        """
        out = self.model(
            input_ids=input_ids.to(self.device),
            past_key_values=cache,
            use_cache=True,
        )
        # (1, L, vocab) -> (vocab,): only the last position can predict the next token.
        return out.logits[0, -1], out.past_key_values
