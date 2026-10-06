from transformers import PreTrainedModel


def _eos_ids(model: PreTrainedModel) -> set[int]:
    eos = model.generation_config.eos_token_id
    return set(eos) if isinstance(eos, list) else set(eos)
