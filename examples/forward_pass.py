"""Run ONE forward pass of the model and check it against Hugging Face.

What this script shows:
    1. Text is turned into token ids (numbers).
    2. The model reads the ids once (a "forward pass") and returns logits:
       a score for every word piece in the vocabulary, at every position.
    3. The highest score at the LAST position is the model's guess for the
       next token.
    4. That guess must equal what Hugging Face's `generate()` produces.
       This is the correctness check we reuse at every step of the project.
"""

import torch
from loguru import logger

from vllm_nano.config import ModelConfig
from vllm_nano.models.loader import load_model_and_tokenizer
from vllm_nano.utils.logging_setup import setup_logging


def main() -> None:
    # Console logging only (log_dir=None means: do not create a log file).
    setup_logging(log_dir=None)

    # Settings: which model, which device (mps/cuda/cpu), which dtype (float32).
    config = ModelConfig()

    # The model is the neural network. The tokenizer converts text <-> token ids.
    model, tokenizer = load_model_and_tokenizer(config)

    # Text -> token ids. "pt" means "give me PyTorch tensors".
    # Result holds `input_ids` (the tokens) and `attention_mask` (1 = real token).
    # Shape of input_ids: (batch=1, seq_len=5).
    inputs = tokenizer("The capital of France is", return_tensors="pt")

    # Tensors are created on the CPU; the model lives on `mps`/`cuda`.
    # Both must be on the same device, so move every tensor in `inputs`.
    inputs = inputs.to(config.resolved_device)

    # FORWARD PASS: run the 5 tokens through all layers once.
    # no_grad = do not track gradients (that is only needed for training),
    # which saves memory and time.
    with torch.no_grad():
        # logits shape: (batch=1, seq_len=5, vocab_size=151936).
        # logits[0, i, :] = scores for "what token comes AFTER position i".
        logits = model(**inputs).logits

    # We only care about what follows the LAST token ("is"), so take
    # batch item 0, position -1 (the last one): a vector of 151936 scores.
    # argmax = index of the highest score = greedy choice of the next token.
    # .item() turns the one-element tensor into a plain Python int.
    next_id = logits[0, -1].argmax().item()

    logger.info("input_ids {}, logits {}", tuple(inputs["input_ids"].shape), tuple(logits.shape))
    # decode: token id -> text. {!r} prints it with quotes so spaces are visible.
    logger.info("our next token: {!r}", tokenizer.decode(next_id))

    # Reference answer from Hugging Face. do_sample=False means greedy
    # (always take the top-scoring token), so it is comparable to our argmax.
    out = model.generate(**inputs, max_new_tokens=1, do_sample=False)
    # generate() returns the prompt tokens PLUS the new one, so the new token
    # is the last element.
    hf_id = out[0, -1].item()
    logger.info("HF next token:  {!r}", tokenizer.decode(hf_id))

    # If this fails, our understanding or our code differs from HF's.
    assert next_id == hf_id, "mismatch with HF generate"


# Run main() only when this file is executed directly (not when imported).
if __name__ == "__main__":
    main()
