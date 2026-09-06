"""Stored-versus-active parameter accounting for the Top-1 MoE model."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from torch import nn


@dataclass(frozen=True, slots=True)
class MoEParameterCounts:
    total: int
    backbone_non_expert: int
    router: int
    experts_stored: int
    experts_active_per_token: int
    active_per_token: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def count_moe_parameters(model: nn.Module, *, num_experts: int) -> MoEParameterCounts:
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    total = router = experts = 0
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        identity = id(parameter)
        if identity in seen:
            continue
        seen.add(identity)
        count = parameter.numel()
        total += count
        if ".ffn.router." in name:
            router += count
        elif ".ffn.experts." in name:
            experts += count
    if experts % num_experts:
        raise RuntimeError("stored expert parameter count is not divisible by expert count")
    active_experts = experts // num_experts
    backbone = total - experts - router
    return MoEParameterCounts(
        total=total,
        backbone_non_expert=backbone,
        router=router,
        experts_stored=experts,
        experts_active_per_token=active_experts,
        active_per_token=backbone + router + active_experts,
    )


__all__ = ["MoEParameterCounts", "count_moe_parameters"]
