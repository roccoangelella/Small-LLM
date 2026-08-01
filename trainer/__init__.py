"""Schema-v2 consumer, trainer, and joint-checkpoint integration."""

from .config import TrainerConfig
from .data import LiveBlockConsumer, PreparedBlockDecoder, SchemaV2ShardReader, TokenBatch
from .engine import (
    StepMetrics,
    TrainerEngine,
    TrainingSession,
    build_adamw,
    generate_token_ids,
    seed_everything,
)
from .identity import canonical_hash, checkpoint_identity, saved_checkpoint_identity
from .schedule import TokenLRScheduler

__all__ = [
    "LiveBlockConsumer",
    "PreparedBlockDecoder",
    "SchemaV2ShardReader",
    "StepMetrics",
    "TokenBatch",
    "TokenLRScheduler",
    "TrainerConfig",
    "TrainerEngine",
    "TrainingSession",
    "build_adamw",
    "canonical_hash",
    "checkpoint_identity",
    "generate_token_ids",
    "saved_checkpoint_identity",
    "seed_everything",
]
