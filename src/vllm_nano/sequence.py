from dataclasses import dataclass
from typing import Literal

# Matches vLLM's vocabulary: "stop" = the model produced EOS, "length" = the token budget ran out.
FinishReason = Literal["stop", "length"]


@dataclass(frozen=True)
class GenerationResult:
    """What one request produced, plus why the loop ended.

    `token_ids` holds only the NEW tokens (the prompt is excluded). The EOS token is
    included when it stopped the loop, so the output matches HF `generate` exactly.
    `finish_reason` is the information the scheduler will need later: an EOS stop can
    free the request's cache, while a length stop means the caller asked for too few
    tokens.
    """

    token_ids: list[int]
    finish_reason: FinishReason
