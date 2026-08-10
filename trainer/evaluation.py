"""Held-out next-token evaluation and checkpoint-generation smoke tests."""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .precision import autocast_context
from .step import _microbatch_to_device, _ordered_batch_tensors
from .types import TokenBatch


@torch.inference_mode()
def evaluate_batches(
    engine: object,
    batches: Iterable[TokenBatch],
    *,
    maximum_batches: int | None = None,
    microbatch_size: int = 1,
) -> dict[str, float | int]:
    """Evaluate held-out blocks with a separately bounded inference microbatch.

    Masked variable-length SFT blocks use the same dynamic microbatch cropping
    as training, so padding to the longest record in an optimizer block is never
    needlessly transferred to the accelerator. Both ``validation`` and ``test``
    are accepted as held-out splits; training blocks are rejected.
    """

    if maximum_batches is not None and maximum_batches <= 0:
        raise ValueError("maximum_batches must be positive")
    if isinstance(microbatch_size, bool) or not isinstance(microbatch_size, int):
        raise ValueError("microbatch_size must be a positive integer")
    if microbatch_size <= 0:
        raise ValueError("microbatch_size must be a positive integer")

    was_training = engine.model.training
    engine.model.eval()

    optimizer = getattr(engine, "optimizer", None)
    zero_grad = getattr(optimizer, "zero_grad", None)
    if callable(zero_grad):
        zero_grad(set_to_none=True)
    if engine.device.type == "cuda":
        torch.cuda.empty_cache()

    total_loss, total_tokens, block_count = 0.0, 0, 0
    observed_split: str | None = None
    try:
        for batch in batches:
            if batch.split not in {"validation", "test"}:
                raise ValueError("evaluation requires validation- or test-split batches")
            if observed_split is None:
                observed_split = batch.split
            elif batch.split != observed_split:
                raise ValueError("one held-out evaluation cannot mix validation and test batches")

            input_ids, labels = _ordered_batch_tensors(batch)
            for start in range(0, batch.sequence_count, microbatch_size):
                stop = min(batch.sequence_count, start + microbatch_size)
                microbatch_inputs, microbatch_labels = _microbatch_to_device(
                    input_ids,
                    labels,
                    start=start,
                    stop=stop,
                    device=engine.device,
                )
                with autocast_context(engine.config.precision, engine.device):
                    logits = engine.model(microbatch_inputs)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        microbatch_labels.reshape(-1),
                        reduction="sum",
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite held-out loss")
                total_loss += float(loss.float())
                total_tokens += int(microbatch_labels.ne(-100).sum().item())
                del logits, loss, microbatch_inputs, microbatch_labels

            block_count += 1
            if maximum_batches is not None and block_count >= maximum_batches:
                break
    finally:
        engine.model.train(was_training)
        if engine.device.type == "cuda":
            torch.cuda.empty_cache()

    if total_tokens == 0:
        raise RuntimeError("held-out source yielded no active targets")
    mean = total_loss / total_tokens
    if observed_split == "validation" and (
        engine.best_validation_loss is None or mean < engine.best_validation_loss
    ):
        engine.best_validation_loss = mean
    return {
        "loss": mean,
        "perplexity": math.exp(min(mean, 80.0)),
        "target_tokens": total_tokens,
        "blocks": block_count,
    }


@torch.no_grad()
def generate_token_ids(
    model: nn.Module,
    prompt: Tensor,
    *,
    max_new_tokens: int,
    max_seq_len: int,
    eos_token_id: int | None = None,
) -> Tensor:
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


__all__ = ["evaluate_batches", "generate_token_ids"]
