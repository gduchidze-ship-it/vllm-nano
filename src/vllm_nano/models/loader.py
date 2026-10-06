from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from vllm_nano.config import ModelConfig


def load_model_and_tokenizer(
    config: ModelConfig,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(
        config.model, trust_remote_code=config.trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        dtype=config.torch_dtype,  # transformers 5 uses `dtype`
        trust_remote_code=config.trust_remote_code,
    )
    return model.to(config.resolved_device).eval(), tokenizer