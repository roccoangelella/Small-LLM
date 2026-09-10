"""Initialization adapter for MoE experts and routers."""

from __future__ import annotations

import math

from model.initialization import initialize_model

from .model import MoESmallLLM


def initialize_moe_model(model: MoESmallLLM, method: str = "normal") -> MoESmallLLM:
    """Use the dense initializer, then restore the frozen small-random routers.

    The dense initializer scales SwiGLU ``down`` projections by the residual factor, but it
    recognises them as ``nn.Linear`` modules. Stacked expert weights are bare parameters, so
    the same factor is applied to them here; without it the two dispatch implementations of
    the same geometry would start from different distributions.
    """
    initialize_model(model, method)
    layers = max(1, len(getattr(model, "blocks", ())))
    model.scale_expert_output_projections(1.0 / math.sqrt(2.0 * layers))
    model.reset_router_parameters()
    return model


__all__ = ["initialize_moe_model"]
