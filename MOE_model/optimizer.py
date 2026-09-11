"""Fail-closed optimizer routing for the MoE branch."""

from __future__ import annotations

import os

from torch import nn
from torch.optim import Optimizer

from model.accounting import optimizer_no_weight_decay_parameter_names
from trainer.config import TrainerConfig
from trainer.optimizer import (
    HybridMuonAdamW,
    OptimizerRouting,
    _ClassifiedParameters,
    _is_known_adamw_parameter,
    _is_muon_parameter,
)
from trainer.optimizer_telemetry import InstrumentedHybridMuonAdamW


_STACKED_EXPERT_SUFFIXES = (".ffn.gate_weight", ".ffn.up_weight", ".ffn.down_weight")


def _is_expert_matrix(name: str, parameter: nn.Parameter) -> bool:
    if name.endswith(_STACKED_EXPERT_SUFFIXES) and parameter.ndim == 3:
        # One stacked parameter per projection: a batch of per-expert matrices.
        return True
    expert_path = ".ffn.experts." in name or name.startswith("ffn.experts.")
    return (
        parameter.ndim == 2
        and expert_path
        and name.endswith((".gate.weight", ".up.weight", ".down.weight"))
    )


def _is_router_matrix(name: str, parameter: nn.Parameter) -> bool:
    return parameter.ndim == 2 and (
        name == "ffn.router.projection.weight"
        or name.endswith(".ffn.router.projection.weight")
    )


def classify_moe_parameters(model: nn.Module) -> _ClassifiedParameters:
    router_without_decay = int(getattr(getattr(model, "config", None), "version", 2)) >= 3
    exclusions = optimizer_no_weight_decay_parameter_names(model)
    muon_names: list[str] = []
    adamw_decay_names: list[str] = []
    adamw_no_decay_names: list[str] = []
    muon: list[nn.Parameter] = []
    adamw_decay: list[nn.Parameter] = []
    adamw_no_decay: list[nn.Parameter] = []
    seen: set[int] = set()

    named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    for name, parameter in named:
        identity = id(parameter)
        if identity in seen:
            raise ValueError(f"trainable parameter is exposed more than once: {name}")
        seen.add(identity)
        if _is_expert_matrix(name, parameter) or _is_muon_parameter(name, parameter):
            muon_names.append(name)
            muon.append(parameter)
        elif _is_router_matrix(name, parameter):
            # The version-3 router contract puts the router matrix on AdamW with
            # weight_decay = 0 and no router-specific learning-rate multiplier. The
            # frozen version-2 identity keeps it in the decaying group.
            if router_without_decay:
                adamw_no_decay_names.append(name)
                adamw_no_decay.append(parameter)
            else:
                adamw_decay_names.append(name)
                adamw_decay.append(parameter)
        elif _is_known_adamw_parameter(name):
            if name in exclusions:
                adamw_no_decay_names.append(name)
                adamw_no_decay.append(parameter)
            else:
                adamw_decay_names.append(name)
                adamw_decay.append(parameter)
        else:
            raise ValueError(
                "MoE trainable parameter has no explicit optimizer route: "
                f"{name} shape={tuple(parameter.shape)}"
            )

    if len(seen) != len(named):
        raise ValueError("MoE optimizer did not classify every trainable parameter")
    if not muon or not adamw_decay:
        raise ValueError("MoE hybrid optimizer requires both Muon and AdamW groups")

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


def build_moe_optimizer(model: nn.Module, config: TrainerConfig) -> Optimizer:
    if config.optimizer != "hybrid_muon_adamw":
        raise ValueError("first MoE experiment is frozen to hybrid Muon + AdamW")
    classified = classify_moe_parameters(model)
    disabled = os.environ.get(
        "SMALL_LLM_DISABLE_OPTIMIZER_TELEMETRY", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    if disabled:
        return HybridMuonAdamW(classified, config)
    return InstrumentedHybridMuonAdamW(classified, config)


__all__ = ["build_moe_optimizer", "classify_moe_parameters"]
