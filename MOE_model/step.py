"""Z-loss-aware atomic optimizer step for the Top-1 MoE experiment."""

from __future__ import annotations

import time

import torch
from torch.nn import functional as F

from trainer.precision import autocast_context
from trainer.step import (
    _clear_optimizer_step_statistics,
    _fp16_overflow_retry_limit,
    _microbatch_to_device,
    _optimizer_gradient_norms,
    _optimizer_step_statistics,
    _ordered_batch_tensors,
)
from trainer.types import TokenBatch

from .metrics import MoEStepMetrics


def _empty_telemetry(engine: object) -> list[dict[str, object]]:
    return [
        {
            "counts": torch.zeros(
                engine.model.config.num_experts,
                dtype=torch.long,
                device=engine.device,
            ),
            "selected_probability_sum": torch.zeros(
                (), dtype=torch.float32, device=engine.device
            ),
            "entropy_sum": torch.zeros(
                (), dtype=torch.float32, device=engine.device
            ),
            "token_count": 0,
        }
        for _ in range(engine.model.config.n_layers)
    ]


def _accumulate_telemetry(
    accumulator: list[dict[str, object]], layers: tuple[object, ...]
) -> None:
    if len(accumulator) != len(layers):
        raise RuntimeError("MoE telemetry layer count mismatch")
    for target, layer in zip(accumulator, layers, strict=True):
        target["counts"].add_(layer.expert_counts)
        target["selected_probability_sum"].add_(layer.selected_probability_sum)
        target["entropy_sum"].add_(layer.entropy_sum)
        target["token_count"] = int(target["token_count"]) + int(layer.token_count)


def _finalize_telemetry(engine: object, accumulator: list[dict[str, object]]) -> dict[str, object]:
    num_experts = int(engine.model.config.num_experts)
    global_counts = torch.zeros(num_experts, dtype=torch.long, device=engine.device)
    layer_payload: dict[str, object] = {}
    dead_slots = 0

    for layer_id, item in enumerate(accumulator):
        counts = item["counts"]
        token_count = int(item["token_count"])
        if token_count <= 0:
            raise RuntimeError("MoE telemetry recorded an empty layer")
        global_counts.add_(counts)
        fractions = counts.float() / float(token_count)
        dead = int((counts == 0).sum().item())
        dead_slots += dead
        layer_payload[f"layer_{layer_id:02d}"] = {
            "expert_counts": [int(v) for v in counts.detach().cpu().tolist()],
            "expert_fractions": [float(v) for v in fractions.detach().cpu().tolist()],
            "max_load_fraction": float(fractions.max().item()),
            "min_load_fraction": float(fractions.min().item()),
            "dead_experts": dead,
            "selected_probability_mean": float(
                item["selected_probability_sum"].item() / token_count
            ),
            "router_entropy_mean": float(item["entropy_sum"].item() / token_count),
        }

    total_assignments = int(global_counts.sum().item())
    global_fractions = global_counts.float() / float(total_assignments)
    accounting = engine.parameter_accounting.as_dict()
    return {
        "routing": {
            "global_expert_counts": [int(v) for v in global_counts.detach().cpu().tolist()],
            "global_expert_fractions": [
                float(v) for v in global_fractions.detach().cpu().tolist()
            ],
            "global_max_load_fraction": float(global_fractions.max().item()),
            "global_min_load_fraction": float(global_fractions.min().item()),
            "dead_layer_expert_slots": int(dead_slots),
            "dropped_tokens": 0,
            "top_k": 1,
            "num_experts": num_experts,
        },
        "layers": layer_payload,
        "parameters": accounting,
    }


def moe_train_step(engine: object, batch: TokenBatch) -> MoEStepMetrics:
    if batch.split != "train" or batch.sequence_count <= 0:
        raise ValueError("training requires a non-empty train-split block")
    started = time.perf_counter()
    if engine.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(engine.device)

    retries = 0
    initial_scaler_scale = float(engine.scaler.get_scale())
    scaler_scale = initial_scaler_scale
    overflow_retry_limit = (
        _fp16_overflow_retry_limit(
            engine.scaler, engine.config.max_overflow_retries
        )
        if engine.scaler.is_enabled()
        else engine.config.max_overflow_retries
    )
    z_coefficient = float(engine.model.config.router_z_loss_coefficient)

    while True:
        engine.optimizer.zero_grad(set_to_none=True)
        next_tokens = engine.consumed_tokens + batch.target_token_count
        lr = engine.scheduler.prepare_step(next_tokens)
        total_ce = torch.zeros((), dtype=torch.float32, device=engine.device)
        total_z = torch.zeros((), dtype=torch.float32, device=engine.device)
        telemetry = _empty_telemetry(engine)

        input_ids, labels = _ordered_batch_tensors(batch)
        size = engine.config.microbatch_size
        total_positions = int(input_ids.numel())
        if total_positions <= 0:
            raise RuntimeError("MoE training block has no input positions")

        for start in range(0, batch.sequence_count, size):
            stop = min(batch.sequence_count, start + size)
            micro_inputs, micro_labels = _microbatch_to_device(
                input_ids,
                labels,
                start=start,
                stop=stop,
                device=engine.device,
            )
            position_fraction = float(micro_inputs.numel()) / float(total_positions)
            with autocast_context(engine.config.precision, engine.device):
                logits, aux = engine.model.forward_with_aux(micro_inputs)
                if logits.ndim != 3 or logits.shape[:2] != micro_labels.shape:
                    raise RuntimeError("MoE logits do not match training labels")
                ce_sum = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    micro_labels.reshape(-1),
                    reduction="sum",
                )
                objective = (
                    ce_sum / batch.target_token_count
                    + z_coefficient * aux.z_loss * position_fraction
                )

            if not bool(torch.isfinite(ce_sum)) or not bool(torch.isfinite(aux.z_loss)):
                engine.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError("non-finite MoE CE or router z-loss")
            total_ce += ce_sum.detach().float()
            total_z += aux.z_loss.detach().float() * position_fraction
            _accumulate_telemetry(telemetry, aux.layers)
            engine.scaler.scale(objective).backward()

        engine.scaler.unscale_(engine.optimizer)
        role_gradient_norms = _optimizer_gradient_norms(engine.optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            engine.model.parameters(), engine.config.max_grad_norm
        )
        finite_gradient = bool(torch.isfinite(gradient_norm))
        if not finite_gradient and not engine.scaler.is_enabled():
            raise FloatingPointError("non-finite MoE gradient norm")
        grad_value = float(gradient_norm.detach())
        gradient_clipped = (
            finite_gradient and grad_value > float(engine.config.max_grad_norm)
        )

        scale_before = float(engine.scaler.get_scale())
        _clear_optimizer_step_statistics(engine.optimizer)
        engine.scaler.step(engine.optimizer)
        engine.scaler.update()
        scaler_scale = float(engine.scaler.get_scale())
        if engine.scaler.is_enabled() and (
            not finite_gradient or scaler_scale < scale_before
        ):
            retries += 1
            engine.overflow_events += 1
            if retries > overflow_retry_limit:
                engine.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    "FP16 MoE optimizer step remained non-finite after dynamic scale calibration"
                )
            continue

        update_statistics = _optimizer_step_statistics(engine.optimizer)
        ce_value = float(total_ce / batch.target_token_count)
        z_value = float(total_z)
        objective_value = ce_value + z_coefficient * z_value
        engine.consumed_tokens = next_tokens
        engine.global_step += 1
        engine.scheduler.commit(engine.consumed_tokens)
        moe_payload = _finalize_telemetry(engine, telemetry)
        break

    elapsed = max(time.perf_counter() - started, 1e-12)
    if engine.device.type == "cuda":
        peak = int(torch.cuda.max_memory_allocated(engine.device))
        peak_reserved = int(torch.cuda.max_memory_reserved(engine.device))
    else:
        peak = peak_reserved = 0

    return MoEStepMetrics(
        step=engine.global_step,
        block_id=batch.block_id,
        loss=ce_value,
        objective_loss=objective_value,
        router_z_loss=z_value,
        router_z_loss_coefficient=z_coefficient,
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
        moe=moe_payload,
    )


__all__ = ["moe_train_step"]
