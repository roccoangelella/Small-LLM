"""Trainer engine adapter for the Top-1 MoE experiment."""

from __future__ import annotations

import torch
from pathlib import Path
from typing import Mapping

from trainer.config import MOE_RESUME_EXECUTION_FIELDS, TrainerConfig
from trainer.engine import TrainerEngine
from trainer.state import load_trainer_state_file

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

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        # A logical block is the optimizer batch. Its microbatch partition and
        # observation cadence may change across single-GPU providers; the
        # scientific recipe, model, optimizer moments and token clock may not.
        if self.model.config.version == 3:
            saved = state.get("config")
            target = self.config.as_dict()
            scheduler = state.get("scheduler")
            if not isinstance(saved, Mapping) or not isinstance(scheduler, Mapping):
                raise ValueError("checkpoint has no trainer/scheduler configuration")
            if set(saved) != set(target):
                raise ValueError("checkpoint trainer configuration fields differ")
            TrainerConfig(**saved)
            if scheduler.get("config") != saved:
                raise ValueError("checkpoint scheduler and trainer configurations disagree")
            changed = {key for key in set(saved) | set(target) if saved.get(key) != target.get(key)}
            if changed - MOE_RESUME_EXECUTION_FIELDS:
                raise ValueError(f"checkpoint scientific configuration mismatch: {sorted(changed - MOE_RESUME_EXECUTION_FIELDS)}")
            state = {**state, "config": target, "scheduler": {**scheduler, "config": target}}
        super().load_state_dict(state)

    def load_checkpoint_state(self, path: Path | str) -> None:
        self.load_state_dict(load_trainer_state_file(path, map_location="cpu"))


__all__ = ["MoETrainerEngine"]
