"""Trainer engine adapter for the Top-1 MoE experiment."""

from __future__ import annotations

import torch

from trainer.config import TrainerConfig
from trainer.engine import TrainerEngine

from .accounting import count_moe_parameters
from .model import MoESmallLLM
from .optimizer import build_moe_optimizer
from .step import moe_train_step


class MoETrainerEngine(TrainerEngine):
    def __init__(
        self,
        model: MoESmallLLM,
        config: TrainerConfig,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        resolved = torch.device(
            device if device is not None else (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        )
        model.to(resolved)
        optimizer = build_moe_optimizer(model, config)
        super().__init__(model, config, device=resolved, optimizer=optimizer)
        self.parameter_accounting = count_moe_parameters(
            model, num_experts=model.config.num_experts, top_k=model.config.top_k
        )

    def train_batch(self, batch: object):
        return moe_train_step(self, batch)


__all__ = ["MoETrainerEngine"]
