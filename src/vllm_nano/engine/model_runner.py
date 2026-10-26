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
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        cache: Cache | None = None,
    ) -> tuple[torch.Tensor, Cache]:
        """Run one forward pass; return the LAST position's logits and the KV cache.

        `input_ids` is (batch, L). The first call passes the whole prompt per row
        (prefill) and returns a cache holding those tokens. Every later call passes
        exactly one token per row (decode): the model reads the past K/V from `cache`
        and appends its own, so a step's cost stops growing with the sequence.

        `attention_mask` and `position_ids` are only needed once a batch is padded;
        for a single unpadded sequence the model derives both itself. Returned logits
        are (batch, vocab) — the caller samples from the last column.
        """
        # Only move what was actually passed: .to(None) is not allowed, and `cache`
        # already lives on the device the model created it on.
        out = self.model(
            input_ids=input_ids.to(self.device),
            attention_mask=None if attention_mask is None else attention_mask.to(self.device),
            position_ids=None if position_ids is None else position_ids.to(self.device),
            past_key_values=cache,
            use_cache=True,
        )
        # (batch, L, vocab) -> (batch, vocab): only the last position predicts the next token.
        return out.logits[:, -1], out.past_key_values
