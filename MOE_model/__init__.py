"""First Small-LLM mixture-of-experts experiment."""

from .accounting import MoEParameterCounts, count_moe_parameters
from .compile_lane import COMPILE_MODES, DEFAULT_COMPILE_MODE, apply_compile_lane
from .config import MoEModelConfig
from .engine import MoETrainerEngine
from .model import DroplessTop1MoE, MoESmallLLM
from .optimizer import build_moe_optimizer, classify_moe_parameters
from .router import SwitchTop1Router

__all__ = [
    "COMPILE_MODES",
    "DEFAULT_COMPILE_MODE",
    "DroplessTop1MoE",
    "MoEModelConfig",
    "MoEParameterCounts",
    "MoESmallLLM",
    "MoETrainerEngine",
    "SwitchTop1Router",
    "apply_compile_lane",
    "build_moe_optimizer",
    "classify_moe_parameters",
    "count_moe_parameters",
]
