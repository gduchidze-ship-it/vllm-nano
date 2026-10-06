import torch


def get_device() -> str:
    """Return the best available device: cuda, then mps, then cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
