"""Pure AdamW and explicit whole-matrix Muon + AdamW optimizers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping

import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer

from model.accounting import optimizer_no_weight_decay_parameter_names

from .config import TrainerConfig

_AGGRESSIVE_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
_STABILIZING_COEFFICIENTS = (2.0, -1.5, 0.5)
_MUON_WEIGHT_SUFFIXES = (
    ".ffn.gate.weight",
    ".ffn.up.weight",
    ".ffn.down.weight",
    # Stacked expert weights: one [experts, in, out] parameter per projection.
    ".ffn.gate_weight",
    ".ffn.up_weight",
    ".ffn.down_weight",
    ".mixer.q_proj.weight",
    ".mixer.k_proj.weight",
    ".mixer.v_proj.weight",
    ".mixer.gate_proj.weight",
    ".mixer.erase_proj.weight",
    ".mixer.write_proj.weight",
    ".mixer.decay_proj.0.weight",
    ".mixer.decay_proj.1.weight",
    ".mixer.output_gate.0.weight",
    ".mixer.output_gate.1.weight",
    ".mixer.out_proj.weight",
)
_ADAMW_STRUCTURED_SUFFIXES = (
    ".mixer.q_conv.weight",
    ".mixer.k_conv.weight",
    ".mixer.v_conv.weight",
)


@dataclass(frozen=True, slots=True)
class OptimizerRouting:
    """Inspectable, exhaustive assignment of trainable parameters."""

    muon: tuple[str, ...]
    adamw_decay: tuple[str, ...]
    adamw_no_decay: tuple[str, ...]

    @property
    def all_names(self) -> tuple[str, ...]:
        return self.muon + self.adamw_decay + self.adamw_no_decay

    def as_dict(self) -> dict[str, object]:
        return {
            "muon": list(self.muon),
            "adamw_decay": list(self.adamw_decay),
            "adamw_no_decay": list(self.adamw_no_decay),
            "counts": {
                "muon": len(self.muon),
                "adamw_decay": len(self.adamw_decay),
                "adamw_no_decay": len(self.adamw_no_decay),
            },
        }


@dataclass(frozen=True, slots=True)
class _ClassifiedParameters:
    routing: OptimizerRouting
    muon: tuple[nn.Parameter, ...]
    adamw_decay: tuple[nn.Parameter, ...]
    adamw_no_decay: tuple[nn.Parameter, ...]


def _is_muon_parameter(name: str, parameter: nn.Parameter) -> bool:
    # A stacked expert weight is a batch of logical matrices, not a tensor of its own:
    # Muon orthogonalises each [in, out] slice independently, exactly as it would if the
    # experts were separate modules.
    return parameter.ndim in (2, 3) and name.endswith(_MUON_WEIGHT_SUFFIXES)


def _is_known_adamw_parameter(name: str) -> bool:
    return (
        name == "token_embedding.weight"
        or name.endswith(_ADAMW_STRUCTURED_SUFFIXES)
        or name.endswith(".bias")
        or name.endswith(".A_log")
        or name.endswith(".dt_bias")
        or ("norm" in name.lower() and name.endswith(".weight"))
    )


def _classify_parameters(model: nn.Module) -> _ClassifiedParameters:
    exclusions = optimizer_no_weight_decay_parameter_names(model)
    named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    names = {name for name, _ in named}
    unknown_exclusions = exclusions - names
    if unknown_exclusions:
        raise ValueError(
            "weight-decay exclusions name missing parameters: "
            f"{sorted(unknown_exclusions)}"
        )

    muon_names: list[str] = []
    adamw_decay_names: list[str] = []
    adamw_no_decay_names: list[str] = []
    muon: list[nn.Parameter] = []
    adamw_decay: list[nn.Parameter] = []
    adamw_no_decay: list[nn.Parameter] = []
    seen_parameters: set[int] = set()

    for name, parameter in named:
        identity = id(parameter)
        if identity in seen_parameters:
            raise ValueError(f"trainable parameter is exposed more than once: {name}")
        seen_parameters.add(identity)

        if _is_muon_parameter(name, parameter):
            muon_names.append(name)
            muon.append(parameter)
        elif _is_known_adamw_parameter(name):
            if name in exclusions:
                adamw_no_decay_names.append(name)
                adamw_no_decay.append(parameter)
            else:
                adamw_decay_names.append(name)
                adamw_decay.append(parameter)
        else:
            raise ValueError(
                "trainable parameter has no explicit Muon/AdamW route: "
                f"{name} shape={tuple(parameter.shape)}"
            )

    if not muon:
        raise ValueError("hybrid Muon + AdamW requires at least one Muon matrix")
    if not adamw_decay:
        raise ValueError("hybrid Muon + AdamW requires an AdamW decay group")
    if len(seen_parameters) != len(named):
        raise ValueError("optimizer routing did not assign every trainable parameter exactly once")

    routing = OptimizerRouting(
        muon=tuple(muon_names),
        adamw_decay=tuple(adamw_decay_names),
        adamw_no_decay=tuple(adamw_no_decay_names),
    )
    return _ClassifiedParameters(
        routing=routing,
        muon=tuple(muon),
        adamw_decay=tuple(adamw_decay),
        adamw_no_decay=tuple(adamw_no_decay),
    )


def optimizer_parameter_routing(model: nn.Module) -> OptimizerRouting:
    """Return the fail-closed parameter-role assignment used by hybrid Muon."""

    return _classify_parameters(model).routing


def build_adamw(model: nn.Module, config: TrainerConfig) -> AdamW:
    """Build the pure-AdamW control using the model decay contract."""

    exclusions = optimizer_no_weight_decay_parameter_names(model)
    named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    names = {name for name, _ in named}
    unknown = exclusions - names
    if unknown:
        raise ValueError(f"weight-decay exclusions name missing parameters: {sorted(unknown)}")
    decay = [parameter for name, parameter in named if name not in exclusions]
    no_decay = [parameter for name, parameter in named if name in exclusions]
    if not decay:
        raise ValueError("AdamW decay parameter group is empty")
    groups: list[dict[str, object]] = [
        {
            "params": decay,
            "weight_decay": config.weight_decay,
            "lr_scale": 1.0,
            "optimizer_role": "adamw_decay",
        }
    ]
    if no_decay:
        groups.append(
            {
                "params": no_decay,
                "weight_decay": 0.0,
                "lr_scale": 1.0,
                "optimizer_role": "adamw_no_decay",
            }
        )
    return AdamW(
        groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.adam_epsilon,
    )


def _newton_schulz_orthogonalize(update: Tensor, *, target_rms: float) -> Tensor:
    """Return a whole-matrix FP32 Muon update with a fixed output RMS."""

    if update.ndim != 2:
        raise ValueError("Muon accepts only complete rank-2 logical matrices")
    value = update.float()
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError("Muon received a non-finite gradient")
    norm = torch.linalg.vector_norm(value)
    if float(norm) == 0.0:
        return torch.zeros_like(value)
    value = value / norm.clamp_min(torch.finfo(torch.float32).eps)

    transposed = value.shape[0] > value.shape[1]
    if transposed:
        value = value.transpose(0, 1)
    coefficients = (
        (_AGGRESSIVE_COEFFICIENTS,) * 8
        + (_STABILIZING_COEFFICIENTS,) * 2
    )
    for a, b, c in coefficients:
        gram = value @ value.transpose(-1, -2)
        value = a * value + (b * gram + c * (gram @ gram)) @ value
    if transposed:
        value = value.transpose(0, 1)

    rms = value.square().mean().sqrt()
    if not bool(torch.isfinite(rms)) or float(rms) == 0.0:
        raise FloatingPointError("Muon Newton-Schulz produced an invalid update RMS")
    value = value * (float(target_rms) / rms)
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError("Muon Newton-Schulz produced non-finite values")
    return value


def _newton_schulz_orthogonalize_batched(updates: Tensor, *, target_rms: float) -> Tensor:
    """Batched whole-matrix Muon update: one Newton-Schulz run for a stack of
    same-shape logical matrices, with per-matrix normalization and output RMS.

    Slice i of the result equals ``_newton_schulz_orthogonalize(updates[i])`` up
    to floating-point summation order. Batching changes the number of kernel
    launches (one per iteration for the whole stack), never the per-matrix
    arithmetic contract: every matrix keeps its own norm, its own zero-update
    short circuit and its own RMS rescale.
    """

    if updates.ndim != 3:
        raise ValueError("batched Muon accepts a stack of complete rank-2 logical matrices")
    value = updates.float()
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError("Muon received a non-finite gradient")
    norms = torch.linalg.vector_norm(value, dim=(-2, -1), keepdim=True)
    nonzero = norms > 0.0
    value = value / norms.clamp_min(torch.finfo(torch.float32).eps)

    transposed = value.shape[-2] > value.shape[-1]
    if transposed:
        value = value.transpose(-2, -1)
    coefficients = (
        (_AGGRESSIVE_COEFFICIENTS,) * 8
        + (_STABILIZING_COEFFICIENTS,) * 2
    )
    for a, b, c in coefficients:
        gram = torch.bmm(value, value.transpose(-2, -1))
        value = a * value + torch.bmm(b * gram + c * torch.bmm(gram, gram), value)
    if transposed:
        value = value.transpose(-2, -1)

    rms = value.square().mean(dim=(-2, -1), keepdim=True).sqrt()
    valid = torch.isfinite(rms) & (rms > 0.0)
    if not bool((valid | ~nonzero).all()):
        raise FloatingPointError("Muon Newton-Schulz produced an invalid update RMS")
    value = value * (float(target_rms) / rms.clamp_min(torch.finfo(torch.float32).tiny))
    value = torch.where(nonzero, value, torch.zeros_like(value))
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError("Muon Newton-Schulz produced non-finite values")
    return value


class HybridMuonAdamW(Optimizer):
    """One optimizer for whole-matrix Muon and AdamW exception groups.

    Updates are in-place, without rollback. A late exception invalidates the live
    state: abort and reload a complete checkpoint rather than retrying this step.
    """

    STATE_VERSION = 1
    RECIPE = "deepseek_v4_whole_matrix_hybrid_ns10"

    def __init__(
        self,
        classified: _ClassifiedParameters,
        config: TrainerConfig,
    ) -> None:
        self.config = config
        groups: list[dict[str, object]] = [
            {
                "params": list(classified.muon),
                "lr": config.learning_rate * config.muon_lr_multiplier,
                "lr_scale": config.muon_lr_multiplier,
                "weight_decay": config.muon_weight_decay,
                "optimizer_role": "muon",
            },
            {
                "params": list(classified.adamw_decay),
                "lr": config.learning_rate,
                "lr_scale": 1.0,
                "weight_decay": config.weight_decay,
                "optimizer_role": "adamw_decay",
            },
        ]
        if classified.adamw_no_decay:
            groups.append(
                {
                    "params": list(classified.adamw_no_decay),
                    "lr": config.learning_rate,
                    "lr_scale": 1.0,
                    "weight_decay": 0.0,
                    "optimizer_role": "adamw_no_decay",
                }
            )
        defaults = {
            "lr": config.learning_rate,
            "weight_decay": 0.0,
            "lr_scale": 1.0,
            "optimizer_role": "unknown",
        }
        super().__init__(groups, defaults)
        self.routing = classified.routing

    def identity(self) -> dict[str, object]:
        return {
            "version": self.STATE_VERSION,
            "optimizer": "hybrid_muon_adamw",
            "recipe": self.RECIPE,
            "nesterov_momentum": self.config.muon_momentum,
            "newton_schulz": {
                "aggressive_steps": 8,
                "aggressive_coefficients": _AGGRESSIVE_COEFFICIENTS,
                "stabilizing_steps": 2,
                "stabilizing_coefficients": _STABILIZING_COEFFICIENTS,
            },
            "target_update_rms": self.config.muon_update_rms,
            "muon_lr_multiplier": self.config.muon_lr_multiplier,
            "muon_weight_decay": self.config.muon_weight_decay,
            "routing": self.routing.as_dict(),
        }

    def state_dict(self) -> dict[str, object]:
        payload = super().state_dict()
        payload["small_llm_optimizer"] = self.identity()
        return payload

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        metadata = state_dict.get("small_llm_optimizer")
        if metadata != self.identity():
            raise ValueError("hybrid Muon optimizer identity or routing mismatch")
        ordinary = dict(state_dict)
        ordinary.pop("small_llm_optimizer", None)
        super().load_state_dict(ordinary)
        # torch.optim.Optimizer casts floating state to the parameter dtype
        # during load. Muon and this AdamW branch deliberately keep their
        # optimizer arithmetic in FP32 even when model parameters are FP16.
        for parameter, parameter_state in self.state.items():
            for key in ("momentum_buffer", "exp_avg", "exp_avg_sq"):
                value = parameter_state.get(key)
                if isinstance(value, Tensor):
                    parameter_state[key] = value.to(
                        device=parameter.device, dtype=torch.float32
                    )

    def _apply_weight_decay(self, parameter: nn.Parameter, *, lr: float, decay: float) -> None:
        factor = 1.0 - lr * decay
        if factor < 0.0:
            raise ValueError("optimizer learning rate times weight decay must not exceed 1")
        if decay:
            parameter.mul_(factor)

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
        self._apply_weight_decay(parameter, lr=lr, decay=decay)
        parameter.add_(update.to(dtype=parameter.dtype), alpha=-lr)

    def _on_muon_update(
        self, parameter: nn.Parameter, *, before: Tensor | None, direction: Tensor
    ) -> None:
        """Hook for subclasses that record per-matrix update statistics."""

    _snapshot_before_muon_update = False

    def _muon_group_step(self, group: Mapping[str, object]) -> None:
        """Apply Muon to a parameter group, batching same-shape logical matrices.

        Every parameter is viewed as a batch of [m, n] matrices: a plain weight is one
        matrix, a stacked expert weight is as many as it has experts. Matrices are bucketed
        by (m, n, device), each bucket runs one batched Newton-Schulz, and momentum and
        weight updates use horizontally fused foreach kernels. The per-matrix contract is
        the one of ``_muon_step``; only the launch count changes. Within a bucket every
        gradient is validated before any state is mutated.
        """

        buckets: dict[tuple[int, int, torch.device], list[nn.Parameter]] = {}
        for parameter in group["params"]:
            gradient = parameter.grad
            if gradient is None:
                continue
            if gradient.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients")
            if parameter.ndim not in (2, 3):
                raise RuntimeError("Muon group contains a non-matrix parameter")
            rows, columns = int(parameter.shape[-2]), int(parameter.shape[-1])
            buckets.setdefault((rows, columns, parameter.device), []).append(parameter)

        beta = float(self.config.muon_momentum)
        lr = float(group["lr"])
        decay = float(group["weight_decay"])
        factor = 1.0 - lr * decay
        if factor < 0.0:
            raise ValueError("optimizer learning rate times weight decay must not exceed 1")
        target_rms = float(self.config.muon_update_rms)

        for (rows, columns, _), parameters in buckets.items():
            grads = [parameter.grad.detach().float() for parameter in parameters]
            stacked_grads = torch.cat([g.reshape(-1, rows, columns) for g in grads])
            if not bool(torch.isfinite(stacked_grads).all()):
                raise FloatingPointError("Muon received a non-finite gradient")
            momenta: list[Tensor] = []
            for parameter, grad in zip(parameters, grads):
                state = self.state[parameter]
                momentum = state.get("momentum_buffer")
                if momentum is None:
                    momentum = torch.zeros_like(grad, dtype=torch.float32)
                    state["momentum_buffer"] = momentum
                if not isinstance(momentum, Tensor) or momentum.dtype != torch.float32:
                    raise RuntimeError("Muon momentum state must be an FP32 tensor")
                momenta.append(momentum)
            torch._foreach_mul_(momenta, beta)
            torch._foreach_add_(momenta, grads)
            # Same expression, same order as the per-matrix path: grad + beta * momentum.
            # Computing beta * momentum first instead would round one ULP differently.
            nesterov = stacked_grads.add(
                torch.cat([momentum.reshape(-1, rows, columns) for momentum in momenta]),
                alpha=beta,
            )
            updates = _newton_schulz_orthogonalize_batched(nesterov, target_rms=target_rms)
            sizes = [parameter.numel() // (rows * columns) for parameter in parameters]
            directions = [
                piece.reshape(parameter.shape)
                for piece, parameter in zip(torch.split(updates, sizes), parameters)
            ]
            befores: list[Tensor | None] = [
                parameter.detach().clone() if self._snapshot_before_muon_update else None
                for parameter in parameters
            ]
            if decay:
                torch._foreach_mul_(list(parameters), factor)
            torch._foreach_add_(
                list(parameters),
                [direction.to(dtype=parameter.dtype)
                 for direction, parameter in zip(directions, parameters)],
                alpha=-lr,
            )
            for parameter, before, direction in zip(parameters, befores, directions):
                self._on_muon_update(parameter, before=before, direction=direction)

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
        self._apply_weight_decay(parameter, lr=lr, decay=decay)
        parameter.add_(update.to(dtype=parameter.dtype), alpha=-lr)

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:
        loss: Tensor | None = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            role = group.get("optimizer_role")
            if role == "muon":
                self._muon_group_step(group)
            elif role in {"adamw_decay", "adamw_no_decay"}:
                for parameter in group["params"]:
                    self._adamw_step(parameter, group)
            else:
                raise RuntimeError(f"unknown hybrid optimizer role: {role!r}")
        return loss


def build_hybrid_muon_adamw(
    model: nn.Module, config: TrainerConfig
) -> HybridMuonAdamW:
    """Build the documented fail-closed whole-matrix Muon + AdamW optimizer."""

    if config.optimizer != "hybrid_muon_adamw":
        raise ValueError(
            "hybrid optimizer construction requires "
            "TrainerConfig(optimizer='hybrid_muon_adamw')"
        )
    return HybridMuonAdamW(_classify_parameters(model), config)


def build_optimizer(model: nn.Module, config: TrainerConfig) -> Optimizer:
    """Build the selected optimizer without silently changing its family."""

    if config.optimizer == "adamw":
        return build_adamw(model, config)
    if config.optimizer == "hybrid_muon_adamw":
        return build_hybrid_muon_adamw(model, config)
    raise ValueError(f"unsupported optimizer: {config.optimizer}")


__all__ = [
    "HybridMuonAdamW",
    "OptimizerRouting",
    "build_adamw",
    "build_hybrid_muon_adamw",
    "build_optimizer",
    "optimizer_parameter_routing",
]
