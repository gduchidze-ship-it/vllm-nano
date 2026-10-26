import torch

# One name, one meaning: used by ModelConfig to turn "float32" into torch.float32.
DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
