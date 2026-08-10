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
from .types import IGNORE_INDEX, TokenBatch


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
    return {role: float(total.sqrt().detach()) for role, total in sorted(totals.items())}


def _clear_optimizer_step_statistics(optimizer: Optimizer) -> None:
    clear = getattr(optimizer, "clear_step_statistics", None)
    if callable(clear):
        clear()


def _optimizer_step_statistics(optimizer: Optimizer) -> dict[str, object]:
    inspect = getattr(optimizer, "step_statistics", None)
    if not callable(inspect):
        return {}
    value = inspect()
    if not isinstance(value, Mapping):
        raise RuntimeError("optimizer step statistics must be a mapping")
    return dict(value)


def _fp16_overflow_retry_limit(scaler: object, configured_retries: int) -> int:
    """Allow enough skipped attempts to calibrate the scale down to one."""

    if isinstance(configured_retries, bool) or not isinstance(configured_retries, int):
        raise TypeError("configured FP16 overflow retries must be an integer")
    if configured_retries < 0:
        raise ValueError("configured FP16 overflow retries must be non-negative")
    get_scale = getattr(scaler, "get_scale", None)
    get_backoff = getattr(scaler, "get_backoff_factor", None)
    if not callable(get_scale) or not callable(get_backoff):
        return configured_retries
    initial_scale = float(get_scale())
    backoff = float(get_backoff())
    if not math.isfinite(initial_scale) or initial_scale <= 0:
        raise FloatingPointError(f"invalid FP16 loss scale: {initial_scale!r}")
    if not math.isfinite(backoff) or not 0.0 < backoff < 1.0:
        raise FloatingPointError(f"invalid FP16 scale backoff factor: {backoff!r}")
    if initial_scale <= 1.0:
        return configured_retries
    reductions_to_one = math.ceil(math.log(1.0 / initial_scale) / math.log(backoff))
    return max(configured_retries, reductions_to_one)


def _ordered_batch_tensors(batch: TokenBatch) -> tuple[torch.Tensor, torch.Tensor]:
    """Length-bucket masked rows locally without changing block membership."""

    inputs, labels = batch.input_ids, batch.labels
    if not bool(labels.eq(IGNORE_INDEX).any()):
        return inputs, labels
    positions = torch.arange(labels.shape[1], dtype=torch.long).unsqueeze(0) + 1
    active_widths = torch.where(labels.ne(IGNORE_INDEX), positions, 0).amax(dim=1)
    if bool(active_widths.eq(0).any()):
        raise RuntimeError("SFT batch contains a row with no active target")
    order = torch.argsort(active_widths, stable=True)
    return inputs.index_select(0, order), labels.index_select(0, order)


def _microbatch_to_device(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    start: int,
    stop: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transfer only the real width of one execution microbatch."""

    inputs = input_ids[start:stop]
    micro_labels = labels[start:stop]
    if bool(micro_labels.eq(IGNORE_INDEX).any()):
        active_columns = micro_labels.ne(IGNORE_INDEX).any(dim=0)
        nonzero = active_columns.nonzero(as_tuple=False)
        if nonzero.numel() == 0:
            raise RuntimeError("microbatch has no active targets")
        width = int(nonzero[-1].item()) + 1
        inputs = inputs[:, :width]
        micro_labels = micro_labels[:, :width]
    return (
        inputs.to(device=device, non_blocking=True),
        micro_labels.to(device=device, non_blocking=True),
    )


def train_step(engine: object, batch: TokenBatch) -> StepMetrics:
    if batch.split != "train" or batch.sequence_count <= 0:
        raise ValueError("training requires a non-empty train-split block")
    started = time.perf_counter()
    if engine.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(engine.device)
    retries, loss_value, grad_value, lr = 0, math.nan, math.nan, math.nan
    initial_scaler_scale = float(engine.scaler.get_scale())
    scaler_scale = initial_scaler_scale
    overflow_retry_limit = (
        _fp16_overflow_retry_limit(engine.scaler, engine.config.max_overflow_retries)
        if engine.scaler.is_enabled()
        else engine.config.max_overflow_retries
    )
    role_gradient_norms: dict[str, float] = {}
    update_statistics: dict[str, object] = {}
    gradient_clipped = False
    while True:
        engine.optimizer.zero_grad(set_to_none=True)
        next_tokens = engine.consumed_tokens + batch.target_token_count
        lr = engine.scheduler.prepare_step(next_tokens)
        total_loss = torch.zeros((), dtype=torch.float32, device=engine.device)
        input_ids, labels = _ordered_batch_tensors(batch)
        size = engine.config.microbatch_size
        for start in range(0, batch.sequence_count, size):
            stop = min(batch.sequence_count, start + size)
            microbatch_inputs, microbatch_labels = _microbatch_to_device(
                input_ids,
                labels,
                start=start,
                stop=stop,
                device=engine.device,
            )
            with autocast_context(engine.config.precision, engine.device):
                logits = engine.model(microbatch_inputs)
                if logits.ndim != 3 or logits.shape[:2] != microbatch_labels.shape:
                    raise RuntimeError("model logits do not match training labels")
                loss_sum = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    microbatch_labels.reshape(-1),
                    reduction="sum",
                )
            if not torch.isfinite(loss_sum):
                engine.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    "non-finite FP16 training loss; loss-scale reduction cannot "
                    f"repair a forward loss (block={batch.block_id})"
                )
            total_loss += loss_sum.detach().float()
            engine.scaler.scale(loss_sum / batch.target_token_count).backward()

        engine.scaler.unscale_(engine.optimizer)
        role_gradient_norms = _optimizer_gradient_norms(engine.optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(engine.model.parameters(), engine.config.max_grad_norm)
        finite_gradient = bool(torch.isfinite(gradient_norm))
        if not finite_gradient and not engine.scaler.is_enabled():
            raise FloatingPointError("non-finite gradient norm")
        grad_value = float(gradient_norm.detach())
        gradient_clipped = finite_gradient and grad_value > float(engine.config.max_grad_norm)
        scale_before = float(engine.scaler.get_scale())
        _clear_optimizer_step_statistics(engine.optimizer)
        engine.scaler.step(engine.optimizer)
        engine.scaler.update()
        scaler_scale = float(engine.scaler.get_scale())
        if engine.scaler.is_enabled() and (not finite_gradient or scaler_scale < scale_before):
            retries, engine.overflow_events = retries + 1, engine.overflow_events + 1
            if retries > overflow_retry_limit:
                engine.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    "FP16 optimizer step remained non-finite after dynamic scale "
                    "calibration; block remains unacknowledged "
                    f"(block={batch.block_id}, attempts={retries}, "
                    f"initial_scale={initial_scaler_scale:g}, "
                    f"current_scale={scaler_scale:g}, retry_limit={overflow_retry_limit})"
                )
            continue
        update_statistics = _optimizer_step_statistics(engine.optimizer)
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
        optimizer_update_statistics=update_statistics,
    )


__all__ = ["_microbatch_to_device", "_ordered_batch_tensors", "train_step"]
