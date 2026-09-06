"""Initialization adapter for MoE experts and routers."""

from __future__ import annotations

from model.initialization import initialize_model

from .model import MoESmallLLM


def initialize_moe_model(model: MoESmallLLM, method: str = "normal") -> MoESmallLLM:
    """Use the dense initializer, then restore the frozen small-random routers."""
    initialize_model(model, method)
    model.reset_router_parameters()
    return model


__all__ = ["initialize_moe_model"]
