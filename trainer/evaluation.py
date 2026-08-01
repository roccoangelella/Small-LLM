"""Held-out next-token evaluation and checkpoint-generation smoke tests."""
from __future__ import annotations
import math
from typing import Iterable
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .precision import autocast_context
from .types import TokenBatch

@torch.no_grad()
def evaluate_batches(engine: object, batches: Iterable[TokenBatch], *,
                     maximum_batches: int | None = None) -> dict[str, float | int]:
    if maximum_batches is not None and maximum_batches <= 0:
        raise ValueError("maximum_batches must be positive")
    was_training = engine.model.training
    engine.model.eval()
    total_loss, total_tokens, block_count = 0.0, 0, 0
    try:
        for batch in batches:
            if batch.split != "validation":
                raise ValueError("evaluation requires validation-split batches")
            input_ids = batch.input_ids.to(device=engine.device, non_blocking=True)
            labels = batch.labels.to(device=engine.device, non_blocking=True)
            with autocast_context(engine.config.precision, engine.device):
                logits = engine.model(input_ids)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1), reduction="sum")
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite validation loss")
            total_loss += float(loss.float())
            total_tokens += labels.numel()
            block_count += 1
            if maximum_batches is not None and block_count >= maximum_batches:
                break
    finally:
        engine.model.train(was_training)
    if total_tokens == 0:
        raise RuntimeError("validation source yielded no tokens")
    mean = total_loss / total_tokens
    if engine.best_validation_loss is None or mean < engine.best_validation_loss:
        engine.best_validation_loss = mean
    return {"loss": mean, "perplexity": math.exp(min(mean, 80.0)),
            "target_tokens": total_tokens, "blocks": block_count}

@torch.no_grad()
def generate_token_ids(model: nn.Module, prompt: Tensor, *, max_new_tokens: int,
                       max_seq_len: int, eos_token_id: int | None = None) -> Tensor:
    if prompt.dtype != torch.long or prompt.ndim != 2 or prompt.shape[0] <= 0:
        raise ValueError("prompt must be a rank-2 torch.long tensor")
    if max_new_tokens < 0 or max_seq_len <= 0:
        raise ValueError("generation lengths are invalid")
    device = next(model.parameters()).device
    output, was_training = prompt.to(device=device), model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            logits = model(output[:, -max_seq_len:])
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            output = torch.cat((output, next_token), dim=1)
            if eos_token_id is not None and bool(torch.all(next_token == eos_token_id)):
                break
    finally:
        model.train(was_training)
    return output
