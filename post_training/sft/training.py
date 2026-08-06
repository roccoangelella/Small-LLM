"""Reusable S0 training-plan construction over the existing trainer engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch import nn

from trainer.engine import TrainerEngine
from trainer.session import TrainingSession

from .config import SFTSchedulePlan, build_s0_trainer_config
from .storage import SFTShardReader


@dataclass(frozen=True, slots=True)
class SFTTrainingPlan:
    dataset_root: Path
    manifest_identity: str
    block_target_counts: tuple[int, ...]
    learning_rate: float
    microbatch_size: int
    precision: str
    seed: int

    @classmethod
    def from_dataset(
        cls,
        dataset_root: Path | str,
        *,
        learning_rate: float = 3e-5,
        microbatch_size: int = 1,
        precision: str = "fp16",
        seed: int = 17,
    ) -> "SFTTrainingPlan":
        reader = SFTShardReader(dataset_root, split="train")
        return cls(
            dataset_root=Path(dataset_root),
            manifest_identity=reader.manifest_identity,
            block_target_counts=reader.block_target_counts,
            learning_rate=learning_rate,
            microbatch_size=microbatch_size,
            precision=precision,
            seed=seed,
        )

    @property
    def schedule(self) -> SFTSchedulePlan:
        return SFTSchedulePlan.from_block_target_counts(self.block_target_counts)

    def build_engine(
        self,
        model: nn.Module,
        *,
        device: str | None = None,
    ) -> TrainerEngine:
        config = build_s0_trainer_config(
            self.schedule,
            microbatch_size=self.microbatch_size,
            precision=self.precision,
            seed=self.seed,
            learning_rate=self.learning_rate,
        )
        return TrainerEngine(model, config, device=device)

    def build_session(
        self,
        model: nn.Module,
        *,
        device: str | None = None,
        verify_checksums: bool = True,
        pad_token_id: int = 50_256,
    ) -> TrainingSession:
        engine = self.build_engine(model, device=device)
        reader = SFTShardReader(
            self.dataset_root,
            split="train",
            verify_checksums=verify_checksums,
            pad_token_id=pad_token_id,
        )
        if reader.manifest_identity != self.manifest_identity:
            raise RuntimeError("SFT dataset changed after training-plan construction")
        return TrainingSession(engine, reader)

    def identity(self) -> dict[str, object]:
        return {
            "training_stage": "sft_s0",
            "dataset_manifest_identity": self.manifest_identity,
            "block_target_counts": list(self.block_target_counts),
            "learning_rate": self.learning_rate,
            "microbatch_size": self.microbatch_size,
            "precision": self.precision,
            "seed": self.seed,
            "optimizer": "hybrid_muon_adamw",
            "loss_objective": "masked_cross_entropy",
            "prompt_loss_weight": 0.0,
            "assistant_loss_weight": 1.0,
            "replay_loss_weight": 1.0,
        }


__all__ = ["SFTTrainingPlan"]
