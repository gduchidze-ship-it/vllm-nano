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
    # The IDE infers `None` from conditional imports inside transformers.
    # noinspection PyTypeChecker
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        config.model, trust_remote_code=config.trust_remote_code
    )
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        config.model,
        dtype=config.torch_dtype,
        trust_remote_code=config.trust_remote_code,
    )
    model.to(config.resolved_device)
    model.eval()
    return model, tokenizer