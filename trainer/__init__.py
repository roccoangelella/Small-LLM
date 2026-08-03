"""Schema-v2 consumer, trainer, and joint-checkpoint integration."""

from .config import TrainerConfig
from .data import LiveBlockConsumer, PreparedBlockDecoder, TokenBatch
from .shards import SchemaV2ShardReader
from .engine import (
    StepMetrics,
    TrainerEngine,
    TrainingSession,
    build_adamw,
    build_optimizer,
    generate_token_ids,
    seed_everything,
)
from .identity import canonical_hash, checkpoint_identity, saved_checkpoint_identity
from .optimizer import (
    HybridMuonAdamW,
    build_hybrid_muon_adamw,
    optimizer_parameter_routing,
)
from .schedule import TokenLRScheduler

__all__ = [
    "HybridMuonAdamW",
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
    "build_hybrid_muon_adamw",
    "build_optimizer",
    "canonical_hash",
    "checkpoint_identity",
    "generate_token_ids",
    "optimizer_parameter_routing",
    "saved_checkpoint_identity",
    "seed_everything",
]
