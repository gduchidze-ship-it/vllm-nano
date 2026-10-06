import torch
from transformers import PreTrainedModel


class ModelRunner:
    """Owns the model and runs forward passes; knows nothing about stopping or scheduling."""

    def __init__(self, model: PreTrainedModel, device: str) -> None:
        self.model = model
        self.device = device

    @torch.inference_mode()  # no gradients on every call, so it cannot be forgotten
    def next_token(self, input_ids: torch.Tensor) -> int:
        """Return the greedy next token id for a (1, seq_len) tensor of token ids."""
        logits = self.model(input_ids=input_ids.to(self.device)).logits
        # Last position of the only sequence; int() copies the result off the GPU.
        return int(logits[0, -1].argmax())
