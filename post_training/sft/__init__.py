"""Modular supervised fine-tuning support for Small LLM."""

from .builder import SFTDatasetBuilder
from .config import SFTDataConfig, SFTSchedulePlan, build_s0_trainer_config
from .filters import S0RecordFilter
from .interpolation import interpolate_state_dicts
from .schema import ChatMessage, ConversationRecord, SFTBlock, TokenizedSFTRecord
from .storage import SFTDatasetWriter, SFTShardReader
from .template import GPT2ChatTemplate, TiktokenGPT2Encoder

__all__ = [
    "ChatMessage",
    "ConversationRecord",
    "GPT2ChatTemplate",
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
    "build_s0_trainer_config",
    "interpolate_state_dicts",
]


def __getattr__(name: str):
    if name == "SFTTrainingPlan":
        from .training import SFTTrainingPlan
        return SFTTrainingPlan
    raise AttributeError(name)
