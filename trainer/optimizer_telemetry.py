"""Derived update statistics for the qualification hybrid optimizer."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Callable, Mapping

import torch
from torch import Tensor, nn

from .config import TrainerConfig
from .optimizer import (
    HybridMuonAdamW,
    _ClassifiedParameters,
    _newton_schulz_orthogonalize,
)


@dataclass(slots=True)
class _UpdateRecord:
    role: str
    name: str
    elements: int
    weight_square_sum: Tensor
    direction_square_sum: Tensor
    effective_update_square_sum: Tensor


class InstrumentedHybridMuonAdamW(HybridMuonAdamW):
    """Hybrid optimizer that exposes non-checkpointed per-step health metrics.

    The pre-LR optimizer direction is measured in FP32.  Effective-update
    statistics use the actual parameter delta after learning rate, decoupled
    weight decay, and the final model-dtype cast.  Only one parameter is cloned
    at a time; the implementation never keeps a full-model telemetry copy.
    """

    def __init__(
        self,
        classified: _ClassifiedParameters,
        config: TrainerConfig,
    ) -> None:
        self._parameter_names: dict[int, str] = {}
        for names, parameters in (
            (classified.routing.muon, classified.muon),
            (classified.routing.adamw_decay, classified.adamw_decay),
            (classified.routing.adamw_no_decay, classified.adamw_no_decay),
        ):
            for name, parameter in zip(names, parameters, strict=True):
                self._parameter_names[id(parameter)] = name
        self._pending_update_records: list[_UpdateRecord] = []
        self._last_step_statistics: dict[str, object] = {}
        super().__init__(classified, config)

    def clear_step_statistics(self) -> None:
        """Discard stale statistics before a GradScaler candidate step."""

        self._pending_update_records = []
        self._last_step_statistics = {}

    def step_statistics(self) -> dict[str, object]:
        """Return a detached primitive snapshot for the last successful step."""

        return copy.deepcopy(self._last_step_statistics)

    def _record_update(
        self,
        parameter: nn.Parameter,
        *,
        before: Tensor,
        role: str,
        direction: Tensor,
    ) -> None:
        name = self._parameter_names.get(id(parameter))
        if name is None:
            raise RuntimeError("optimizer telemetry found an unnamed parameter")
        weight = before.detach().float()
        update = direction.detach().float()
        if weight.shape != update.shape or parameter.shape != update.shape:
            raise RuntimeError("optimizer telemetry update shape does not match parameter")
        applied_delta = parameter.detach().float() - weight
        self._pending_update_records.append(
            _UpdateRecord(
                role=role,
                name=name,
                elements=parameter.numel(),
                weight_square_sum=weight.square().sum(),
                direction_square_sum=update.square().sum(),
                effective_update_square_sum=applied_delta.square().sum(),
            )
        )

    def _finalize_statistics(self) -> dict[str, object]:
        if not self._pending_update_records:
            return {}
        values = torch.stack(
            [
                torch.stack(
                    (
                        record.weight_square_sum,
                        record.direction_square_sum,
                        record.effective_update_square_sum,
                    )
                )
                for record in self._pending_update_records
            ]
        ).detach().cpu().tolist()

        totals: dict[str, dict[str, float | int]] = {}
        matrix_direction_rms: dict[str, float] = {}
        matrix_effective_ratios: dict[str, float] = {}
        epsilon = float(torch.finfo(torch.float32).eps)

        for record, (weight_sq, direction_sq, effective_sq) in zip(
            self._pending_update_records, values, strict=True
        ):
            role = record.role
            aggregate = totals.setdefault(
                role,
                {
                    "parameter_tensors": 0,
                    "elements": 0,
                    "weight_square_sum": 0.0,
                    "direction_square_sum": 0.0,
                    "effective_update_square_sum": 0.0,
                },
            )
            aggregate["parameter_tensors"] = int(aggregate["parameter_tensors"]) + 1
            aggregate["elements"] = int(aggregate["elements"]) + record.elements
            aggregate["weight_square_sum"] = float(aggregate["weight_square_sum"]) + float(weight_sq)
            aggregate["direction_square_sum"] = float(aggregate["direction_square_sum"]) + float(direction_sq)
            aggregate["effective_update_square_sum"] = float(
                aggregate["effective_update_square_sum"]
            ) + float(effective_sq)

            if role == "muon":
                weight_rms = math.sqrt(max(float(weight_sq), 0.0) / record.elements)
                direction_rms = math.sqrt(max(float(direction_sq), 0.0) / record.elements)
                effective_rms = math.sqrt(max(float(effective_sq), 0.0) / record.elements)
                matrix_direction_rms[record.name] = direction_rms
                matrix_effective_ratios[record.name] = effective_rms / max(
                    weight_rms, epsilon
                )

        result: dict[str, object] = {}
        for role, aggregate in sorted(totals.items()):
            elements = int(aggregate["elements"])
            weight_rms = math.sqrt(
                max(float(aggregate["weight_square_sum"]), 0.0) / elements
            )
            direction_rms = math.sqrt(
                max(float(aggregate["direction_square_sum"]), 0.0) / elements
            )
            effective_rms = math.sqrt(
                max(float(aggregate["effective_update_square_sum"]), 0.0) / elements
            )
            role_result: dict[str, object] = {
                "parameter_tensors": int(aggregate["parameter_tensors"]),
                "elements": elements,
                "pre_update_weight_rms": weight_rms,
                "optimizer_direction_rms": direction_rms,
                "effective_update_rms": effective_rms,
                "effective_update_to_weight_ratio": effective_rms
                / max(weight_rms, epsilon),
            }
            if role == "muon":
                role_result["matrix_optimizer_direction_rms"] = dict(
                    sorted(matrix_direction_rms.items())
                )
                role_result["matrix_effective_update_to_weight_ratio"] = dict(
                    sorted(matrix_effective_ratios.items())
                )
            result[role] = role_result
        return result

    def _muon_step(self, parameter: nn.Parameter, group: Mapping[str, object]) -> None:
        gradient = parameter.grad
        if gradient is None:
            return
        if gradient.is_sparse:
            raise RuntimeError("Muon does not support sparse gradients")
        if parameter.ndim != 2:
            raise RuntimeError("Muon group contains a non-matrix parameter")

        grad = gradient.detach().float()
        if not bool(torch.isfinite(grad).all()):
            raise FloatingPointError("Muon received a non-finite gradient")
        state = self.state[parameter]
        momentum = state.get("momentum_buffer")
        if momentum is None:
            momentum = torch.zeros_like(grad, dtype=torch.float32)
            state["momentum_buffer"] = momentum
        if not isinstance(momentum, Tensor) or momentum.dtype != torch.float32:
            raise RuntimeError("Muon momentum state must be an FP32 tensor")

        beta = float(self.config.muon_momentum)
        momentum.mul_(beta).add_(grad)
        nesterov = grad.add(momentum, alpha=beta)
        update = _newton_schulz_orthogonalize(
            nesterov, target_rms=self.config.muon_update_rms
        )
        lr = float(group["lr"])
        decay = float(group["weight_decay"])
        before = parameter.detach().clone()
        self._apply_weight_decay(parameter, lr=lr, decay=decay)
        parameter.add_(update.to(dtype=parameter.dtype), alpha=-lr)
        self._record_update(
            parameter,
            before=before,
            role="muon",
            direction=update,
        )

    def _adamw_step(self, parameter: nn.Parameter, group: Mapping[str, object]) -> None:
        gradient = parameter.grad
        if gradient is None:
            return
        if gradient.is_sparse:
            raise RuntimeError("AdamW does not support sparse gradients")

        grad = gradient.detach().float()
        if not bool(torch.isfinite(grad).all()):
            raise FloatingPointError("AdamW received a non-finite gradient")
        state = self.state[parameter]
        step = state.get("step", 0)
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise RuntimeError("AdamW step state is invalid")
        step += 1
        state["step"] = step

        exp_avg = state.get("exp_avg")
        exp_avg_sq = state.get("exp_avg_sq")
        if exp_avg is None:
            exp_avg = torch.zeros_like(grad, dtype=torch.float32)
            exp_avg_sq = torch.zeros_like(grad, dtype=torch.float32)
            state["exp_avg"] = exp_avg
            state["exp_avg_sq"] = exp_avg_sq
        if (
            not isinstance(exp_avg, Tensor)
            or not isinstance(exp_avg_sq, Tensor)
            or exp_avg.dtype != torch.float32
            or exp_avg_sq.dtype != torch.float32
        ):
            raise RuntimeError("AdamW moment state must use FP32 tensors")

        beta1, beta2 = float(self.config.beta1), float(self.config.beta2)
        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        bias_correction1 = 1.0 - beta1**step
        bias_correction2 = 1.0 - beta2**step
        denominator = (exp_avg_sq / bias_correction2).sqrt().add_(
            float(self.config.adam_epsilon)
        )
        update = (exp_avg / bias_correction1) / denominator

        lr = float(group["lr"])
        decay = float(group["weight_decay"])
        role = str(group["optimizer_role"])
        before = parameter.detach().clone()
        self._apply_weight_decay(parameter, lr=lr, decay=decay)
        parameter.add_(update.to(dtype=parameter.dtype), alpha=-lr)
        self._record_update(
            parameter,
            before=before,
            role=role,
            direction=update,
        )

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:
        self.clear_step_statistics()
        loss: Tensor | None = None
        try:
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()

            for group in self.param_groups:
                role = group.get("optimizer_role")
                if role == "muon":
                    for parameter in group["params"]:
                        self._muon_step(parameter, group)
                elif role in {"adamw_decay", "adamw_no_decay"}:
                    for parameter in group["params"]:
                        self._adamw_step(parameter, group)
                else:
                    raise RuntimeError(f"unknown hybrid optimizer role: {role!r}")
            self._last_step_statistics = self._finalize_statistics()
            self._pending_update_records = []
            return loss
        except BaseException:
            self.clear_step_statistics()
            raise


__all__ = ["InstrumentedHybridMuonAdamW"]
