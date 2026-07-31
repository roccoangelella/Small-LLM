"""Reusable PyTorch building blocks for the Small LLM."""

from .components import (
    CausalMHA,
    CausalSelfAttention,
    GatedMultiheadAttention,
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
    TiedEmbedding,
    TiedEmbeddingOutput,
    TiedTokenEmbedding,
    apply_rope,
    apply_rope_qk,
)
from .config import ModelConfig, smoke_config, substantive_config

__all__ = [
    "CausalMHA",
    "CausalSelfAttention",
    "GatedMultiheadAttention",
    "ModelConfig",
    "RMSNorm",
    "RotaryEmbedding",
    "SwiGLU",
    "TiedEmbedding",
    "TiedEmbeddingOutput",
    "TiedTokenEmbedding",
    "apply_rope",
    "apply_rope_qk",
    "smoke_config",
    "substantive_config",
]
