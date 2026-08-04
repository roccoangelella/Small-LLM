"""One atomic prepared-block optimizer update."""

from __future__ import annotations

import math
import time
from typing import Mapping

import torch
from torch.nn import functional as F
from torch.optim import Optimizer

from .metrics import StepMetrics
from .precision import autocast_context
from .types import TokenBatch


def _optimizer_gradient_norms(optimizer: Optimizer) -> dict[str, float]:
    """Return pre-clipping L2 gradient norms grouped by optimizer role."""

    totals: dict[str, torch.Tensor] = {}
    for group in optimizer.param_groups:
        role = str(group.get("optimizer_role", "unknown"))
        for parameter in group["params"]:
            gradient = parameter.grad
            if gradient is None:
                continue
            contribution = gradient.detach().float().square().sum()
            totals[role] = contribution if role not in totals else totals[role] + contribution
    return {
        role: float(total.sqrt().detach())
        for role, total in sorted(totals.items())
    }


def train_step(engine: object, batch: TokenBatch) -> StepMetrics:
    if batch.split != "train" or batch.sequence_count <= 0:
        raise ValueError("training requires a non-empty train-split block")
    started = time.perf_counter()
    if engine.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(engine.device)
    retries, loss_value, grad_value, lr = 0, math.nan, math.nan, math.nan
    scaler_scale = float(engine.scaler.get_scale())
    role_gradient_norms: dict[str, float] = {}
    gradient_clipped = False
    while True:
        engine.optimizer.zero_grad(set_to_none=True)
        next_tokens = engine.consumed_tokens + batch.target_token_count
        lr = engine.scheduler.prepare_step(next_tokens)
        total_loss = torch.zeros((), dtype=torch.float32, device=engine.device)
        input_ids = batch.input_ids.to(device=engine.device, non_blocking=True)
        labels = batch.labels.to(device=engine.device, non_blocking=True)
        loss_overflow = False
        size = engine.config.microbatch_size
        for start in range(0, batch.sequence_count, size):
            stop = min(batch.sequence_count, start + size)
            with autocast_context(engine.config.precision, engine.device):
                logits = engine.model(input_ids[start:stop])
                if logits.ndim != 3 or logits.shape[:2] != labels[start:stop].shape:
                    raise RuntimeError("model logits do not match training labels")
                loss_sum = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels[start:stop].reshape(-1),
                    reduction="sum",
                )
            if not torch.isfinite(loss_sum):
                if not engine.scaler.is_enabled():
                    raise FloatingPointError("non-finite training loss")
                retries, engine.overflow_events = retries + 1, engine.overflow_events + 1
                if retries > engine.config.max_overflow_retries:
                    raise FloatingPointError(
                        "FP16 loss repeatedly overflowed; block remains unacknowledged"
                    )
                engine.scaler.update(max(engine.scaler.get_scale() / 2.0, 1.0))
                loss_overflow = True
                break
            total_loss += loss_sum.detach().float()
            engine.scaler.scale(loss_sum / batch.target_token_count).backward()
        if loss_overflow:
            engine.optimizer.zero_grad(set_to_none=True)
            continue

        engine.scaler.unscale_(engine.optimizer)
        role_gradient_norms = _optimizer_gradient_norms(engine.optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            engine.model.parameters(), engine.config.max_grad_norm
        )
        finite_gradient = bool(torch.isfinite(gradient_norm))
        if not finite_gradient and not engine.scaler.is_enabled():
            raise FloatingPointError("non-finite gradient norm")
        grad_value = float(gradient_norm.detach())
        gradient_clipped = finite_gradient and grad_value > float(engine.config.max_grad_norm)
        scale_before = engine.scaler.get_scale()
        engine.scaler.step(engine.optimizer)
        engine.scaler.update()
        scaler_scale = float(engine.scaler.get_scale())
        if engine.scaler.is_enabled() and (
            not finite_gradient or engine.scaler.get_scale() < scale_before
        ):
            retries, engine.overflow_events = retries + 1, engine.overflow_events + 1
            if retries > engine.config.max_overflow_retries:
                engine.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    "FP16 optimizer step repeatedly overflowed; block remains unacknowledged"
                )
            continue
        loss_value = float(total_loss / batch.target_token_count)
        engine.consumed_tokens, engine.global_step = next_tokens, engine.global_step + 1
        engine.scheduler.commit(engine.consumed_tokens)
        break

    elapsed = max(time.perf_counter() - started, 1e-12)
    if engine.device.type == "cuda":
        peak = int(torch.cuda.max_memory_allocated(engine.device))
        peak_reserved = int(torch.cuda.max_memory_reserved(engine.device))
    else:
        peak = 0
        peak_reserved = 0
    return StepMetrics(
        step=engine.global_step,
        block_id=batch.block_id,
        loss=loss_value,
        learning_rate=lr,
        gradient_norm=grad_value,
        sequences=batch.sequence_count,
        target_tokens=batch.target_token_count,
        consumed_tokens=engine.consumed_tokens,
        elapsed_seconds=elapsed,
        tokens_per_second=batch.target_token_count / elapsed,
        overflow_retries=retries,
        peak_memory_bytes=peak,
        grad_scaler_scale=scaler_scale,
        gradient_clipped=gradient_clipped,
        overflow_events_total=engine.overflow_events,
        peak_reserved_memory_bytes=peak_reserved,
        optimizer_gradient_norms=role_gradient_norms,
    )
