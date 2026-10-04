import torch


def get_default_device():
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype(device):
    # bfloat16 is only reliable on CUDA; MPS works best with float16, CPU with float32
    if device.startswith("cuda"):
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32
