"""Modular supervised fine-tuning support for Small LLM."""

from .behavior_eval import BEHAVIOR_CASES, BehaviorCase, evaluate_behavior
from .builder import SFTDatasetBuilder
from .bundle import (
    IdentitySplitPolicy,
    build_bundle,
    prepare_smoltalk,
    sft_budget_from_parent,
    verify_bundle,
)
from .config import SFTDataConfig, SFTSchedulePlan, build_s0_trainer_config
from .filters import S0RecordFilter
from .interpolation import interpolate_state_dicts
from .schema import ChatMessage, ConversationRecord, SFTBlock, TokenizedSFTRecord
from .storage import SFTDatasetWriter, SFTShardReader
from .template import GPT2ChatTemplate, TiktokenGPT2Encoder

__all__ = [
    "BEHAVIOR_CASES",
    "BehaviorCase",
    "ChatMessage",
    "ConversationRecord",
    "GPT2ChatTemplate",
    "IdentitySplitPolicy",
    "S0RecordFilter",
    "SFTBlock",
    "SFTDataConfig",
    "SFTDatasetBuilder",
    "SFTDatasetWriter",
    "SFTSchedulePlan",
    "SFTShardReader",
    "SFTTrainingPlan",
    "TiktokenGPT2Encoder",
    "TokenizedSFTRecord",
    "build_bundle",
    "build_s0_trainer_config",
    "evaluate_behavior",
    "interpolate_state_dicts",
    "prepare_smoltalk",
    "sft_budget_from_parent",
    "verify_bundle",
]


def __getattr__(name: str):
    if name == "SFTTrainingPlan":
        from .training import SFTTrainingPlan

        return SFTTrainingPlan
    raise AttributeError(name)
