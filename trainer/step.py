"""One atomic prepared-block optimizer update."""
from __future__ import annotations
import math, time
import torch
from torch.nn import functional as F
from .metrics import StepMetrics
from .precision import autocast_context
from .types import TokenBatch

def train_step(engine: object, batch: TokenBatch) -> StepMetrics:
    if batch.split != "train" or batch.sequence_count <= 0:
        raise ValueError("training requires a non-empty train-split block")
    started = time.perf_counter()
    if engine.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(engine.device)
    retries, loss_value, grad_value, lr = 0, math.nan, math.nan, math.nan
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
                loss_sum = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                    labels[start:stop].reshape(-1), reduction="sum")
            if not torch.isfinite(loss_sum):
                if not engine.scaler.is_enabled():
                    raise FloatingPointError("non-finite training loss")
                retries, engine.overflow_events = retries + 1, engine.overflow_events + 1
                if retries > engine.config.max_overflow_retries:
                    raise FloatingPointError("FP16 loss repeatedly overflowed; block remains unacknowledged")
                engine.scaler.update(max(engine.scaler.get_scale() / 2.0, 1.0))
                loss_overflow = True
                break
            total_loss += loss_sum.detach().float()
            engine.scaler.scale(loss_sum / batch.target_token_count).backward()
        if loss_overflow:
            engine.optimizer.zero_grad(set_to_none=True)
            continue
        engine.scaler.unscale_(engine.optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(engine.model.parameters(), engine.config.max_grad_norm)
        finite_gradient = bool(torch.isfinite(gradient_norm))
        if not finite_gradient and not engine.scaler.is_enabled():
            raise FloatingPointError("non-finite gradient norm")
        grad_value = float(gradient_norm.detach())
        scale_before = engine.scaler.get_scale()
        engine.scaler.step(engine.optimizer)
        engine.scaler.update()
        if engine.scaler.is_enabled() and (not finite_gradient or engine.scaler.get_scale() < scale_before):
            retries, engine.overflow_events = retries + 1, engine.overflow_events + 1
            if retries > engine.config.max_overflow_retries:
                engine.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError("FP16 optimizer step repeatedly overflowed; block remains unacknowledged")
            continue
        loss_value = float(total_loss / batch.target_token_count)
        engine.consumed_tokens, engine.global_step = next_tokens, engine.global_step + 1
        engine.scheduler.commit(engine.consumed_tokens)
        break
    elapsed = max(time.perf_counter() - started, 1e-12)
    peak = int(torch.cuda.max_memory_allocated(engine.device)) if engine.device.type == "cuda" else 0
    return StepMetrics(engine.global_step, batch.block_id, loss_value, lr, grad_value,
        batch.sequence_count, batch.target_token_count, engine.consumed_tokens, elapsed,
        batch.target_token_count / elapsed, retries, peak)
