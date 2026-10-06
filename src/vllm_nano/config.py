from dataclasses import dataclass, field

import torch
from transformers import AutoConfig, PretrainedConfig

from vllm_nano.constants import _DTYPES
from vllm_nano.utils.device import get_device


@dataclass(frozen=True)
class ModelConfig:
    """
    User-facing model settings plus values resolved from the HF config.
    """

    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dtype: str = "auto"
    device: str = "auto"
    max_model_len: int | None = None
    trust_remote_code: bool = False

    hf_config: PretrainedConfig = field(init=False, repr=False, compare=False)
    torch_dtype: torch.dtype = field(init=False)
    resolved_device: str = field(init=False)
    resolved_max_model_len: int = field(init=False)

    def __post_init__(self) -> None:
        hf_config = AutoConfig.from_pretrained(
            self.model, trust_remote_code=self.trust_remote_code
        )
        device = get_device() if self.device == "auto" else self.device
        # float32 by default so greedy output matches HF token for token.
        dtype_name = "float32" if self.dtype == "auto" else self.dtype
        if dtype_name not in _DTYPES:
            raise ValueError(f"Unsupported dtype {self.dtype!r}; use one of {list(_DTYPES)} or 'auto'")
        model_limit = hf_config.max_position_embeddings
        max_len = self.max_model_len or model_limit
        if max_len > model_limit:
            raise ValueError(f"max_model_len {max_len} exceeds model limit {model_limit}")

        object.__setattr__(self, "hf_config", hf_config)
        object.__setattr__(self, "torch_dtype", _DTYPES[dtype_name])
        object.__setattr__(self, "resolved_device", device)
        object.__setattr__(self, "resolved_max_model_len", max_len)

    @property
    def num_layers(self) -> int:
        return self.hf_config.num_hidden_layers

    @property
    def num_attention_heads(self) -> int:
        return self.hf_config.num_attention_heads

    @property
    def num_kv_heads(self) -> int:
        # Models without GQA only have num_attention_heads.
        return getattr(self.hf_config, "num_key_value_heads", None) or self.num_attention_heads

    @property
    def head_dim(self) -> int:
        return getattr(self.hf_config, "head_dim", None) or (
            self.hf_config.hidden_size // self.num_attention_heads
        )
