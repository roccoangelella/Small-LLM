"""Precision contexts shared by training and evaluation."""
from contextlib import nullcontext
import torch

def autocast_context(precision: str, device: torch.device):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)
