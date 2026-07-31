"""Assembly of the decoder-only Small LLM.

The mixer implementations live in separate modules.  This file deliberately
keeps the assembly layer boring: a decoder block is two sequential pre-norm
residual branches, and the language-model matrix is shared with the token
embedding.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor, nn

from .config import ModelConfig
from .components import GatedMultiheadAttention, RMSNorm, SwiGLU, TiedEmbedding
from .gdn2 import GDN2Cache, GatedDeltaNet2


def _layer_kinds(config: ModelConfig) -> tuple[str, ...]:
    kinds = getattr(config, "layer_kinds", None)
    if kinds is None:
        kinds = config.layer_pattern
    normalized = tuple(str(kind).lower() for kind in kinds)
    if len(normalized) != config.n_layers:
        raise ValueError("config layer kinds must contain exactly n_layers entries")
    if any(kind not in {"gdn", "gdn-2", "gdn_v1", "swa", "mha"} for kind in normalized):
        raise ValueError("layer kinds must be gdn, gdn-2, gdn_v1, swa, or mha")
    return normalized


def _matched_all_mha_ffn_width(config: ModelConfig) -> int:
    """Return the closest integral FFN width for the all-MHA comparison.

    Replacing each GDN-2 mixer with MHA removes parameters.  The baseline
    widens only its SwiGLU branches to compensate; the frozen hybrid geometry
    is never changed.  Integer matrix dimensions prevent literal equality,
    so this minimizes the remaining difference (14,520 parameters at the
    substantive geometry, or 0.015%).
    """

    d_model = config.d_model
    gdn_per_layer = (
        6 * d_model * d_model
        + 3 * d_model * config.gdn_conv_kernel_size
        + 4 * d_model * 64
        + d_model
        + config.gdn_value_dim
        + config.gdn_num_key_heads
        + config.gdn_num_key_heads * config.gdn_key_dim
    )
    mha_per_layer = 5 * d_model * d_model + 2 * config.head_dim
    # Plan C's selected layer kinds are already all MHA.  Its compensation
    # therefore derives from the frozen primary 3:1 pattern it replaces.
    replaced_layers = sum(kind == "gdn" for kind in config.layer_pattern) * (
        config.n_layers // len(config.layer_pattern)
    )
    gap = replaced_layers * (gdn_per_layer - mha_per_layer)
    per_width = config.n_layers * 3 * d_model
    return max(1, config.d_ff + round(gap / per_width))


@dataclass
class ModelCache:
    """Explicit placeholder for a future unified generation cache.

    GDN-2 and full MHA have different state contracts.  Until both contracts
    can be advanced atomically, ``SmallLLM`` refuses a cache rather than
    silently treating a full prefix as a one-token decode.
    """

    gdn_states: tuple[Any, ...] = ()
    mha_states: tuple[Any, ...] = ()
    sequence_offset: int = 0


class DecoderBlock(nn.Module):
    """One sequential pre-norm mixer/FFN decoder block."""

    def __init__(self, config: ModelConfig, mixer: nn.Module, *, d_ff: int | None = None) -> None:
        super().__init__()
        self.mixer = mixer
        self.mixer_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.ffn = SwiGLU(config.d_model, config.d_ff if d_ff is None else d_ff)

    def forward(self, x: Tensor) -> Tensor:
        mixed = self.mixer(self.mixer_norm(x))
        # A reference GDN implementation may return its disposable training
        # state alongside the activations.  The assembly path intentionally
        # ignores that state; it is not a generation cache.
        if isinstance(mixed, tuple):
            mixed = mixed[0]
        x = x + mixed
        return x + self.ffn(self.ffn_norm(x))


class SmallLLM(nn.Module):
    """Hybrid GDN-2/full-MHA decoder with a semantic-vocabulary output."""

    def __init__(self, config: ModelConfig, all_mha: bool = False) -> None:
        super().__init__()
        self.config = config
        self.all_mha = bool(all_mha or config.architecture == "all_mha")
        configured_kinds = _layer_kinds(config)
        self.layer_kinds = tuple("mha" if self.all_mha else kind for kind in configured_kinds)
        self.ffn_width = _matched_all_mha_ffn_width(config) if self.all_mha else config.d_ff

        self.token_embedding = TiedEmbedding(config)
        blocks: list[DecoderBlock] = []
        for kind in self.layer_kinds:
            if kind in {"gdn", "gdn-2"}:
                mixer = GatedDeltaNet2(config)
            elif kind == "gdn_v1":
                raise NotImplementedError(
                    "Plan A.5 requires a separately qualified GDN-v1 backend; "
                    "it must not silently reuse GDN-2"
                )
            elif kind == "swa":
                mixer = GatedMultiheadAttention(replace(config, attention_window=512))
            else:
                mixer = GatedMultiheadAttention(replace(config, attention_window=None))
            blocks.append(DecoderBlock(config, mixer, d_ff=self.ffn_width))
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps)

    @property
    def embedding(self) -> nn.Module:
        """Compatibility spelling for the tied token embedding module."""

        return self.token_embedding

    @property
    def layers(self) -> nn.ModuleList:
        """Compatibility spelling for the decoder block list."""

        return self.blocks

    @property
    def lm_head(self) -> nn.Module:
        """The output is the embedding module's tied projection."""

        return self.token_embedding

    def _logits(self, hidden: Tensor) -> Tensor:
        semantic = self.config.semantic_vocab_size
        for name in ("logits", "lm_head", "project"):
            projection = getattr(self.token_embedding, name, None)
            if callable(projection):
                logits = projection(hidden)
                if logits.shape[-1] > semantic:
                    logits = logits.narrow(-1, 0, semantic)
                if logits.shape[-1] != semantic:
                    raise ValueError("tied embedding projection returned the wrong vocabulary width")
                return logits
        weight = getattr(self.token_embedding, "weight", None)
        if weight is None:
            raise AttributeError("tied embedding must expose logits(hidden) or weight")
        return torch.nn.functional.linear(hidden, weight[:semantic])

    def forward(self, input_ids: Tensor, cache: ModelCache | None = None) -> Tensor:
        if cache is not None:
            raise NotImplementedError(
                "unified cached decoding is not implemented until MHA and GDN-2 "
                "state advancement share one cache contract"
            )
        hidden = self.token_embedding(input_ids)
        for block in self.blocks:
            hidden = block(hidden)
        return self._logits(self.final_norm(hidden))


__all__ = ["DecoderBlock", "GDN2Cache", "ModelCache", "SmallLLM"]
