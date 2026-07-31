"""Validated, immutable geometry for the shared model components."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

LayerKind = Literal["gdn", "mha"]
_DEFAULT_LAYER_PATTERN: tuple[LayerKind, ...] = ("gdn", "gdn", "gdn", "mha")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Geometry shared by the initial model and its small test variants."""

    semantic_vocab_size: int = 50_257
    padded_vocab_size: int = 50_304
    max_seq_len: int = 2_048
    d_model: int = 512
    n_layers: int = 20
    d_ff: int = 1_408
    n_heads: int = 8
    head_dim: int = 64
    gdn_num_key_heads: int = 8
    gdn_num_value_heads: int = 8
    gdn_key_dim: int = 64
    gdn_value_dim: int = 64
    gdn_conv_kernel_size: int = 4
    layer_pattern: tuple[LayerKind, ...] = _DEFAULT_LAYER_PATTERN
    rms_norm_eps: float = 1e-6
    rope_base: float = 10_000.0
    attention_window: int | None = None
    dropout: float = 0.0

    def __post_init__(self) -> None:
        positive_ints = {
            "semantic_vocab_size": self.semantic_vocab_size,
            "padded_vocab_size": self.padded_vocab_size,
            "max_seq_len": self.max_seq_len,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "n_heads": self.n_heads,
            "head_dim": self.head_dim,
            "gdn_num_key_heads": self.gdn_num_key_heads,
            "gdn_num_value_heads": self.gdn_num_value_heads,
            "gdn_key_dim": self.gdn_key_dim,
            "gdn_value_dim": self.gdn_value_dim,
            "gdn_conv_kernel_size": self.gdn_conv_kernel_size,
        }
        for name, value in positive_ints.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

        if self.semantic_vocab_size > self.padded_vocab_size:
            raise ValueError("padded_vocab_size must be at least semantic_vocab_size")
        if self.padded_vocab_size % 8:
            raise ValueError("padded_vocab_size must be divisible by 8")
        if self.d_model != self.n_heads * self.head_dim:
            raise ValueError("d_model must equal n_heads * head_dim")
        if self.gdn_num_key_heads != self.gdn_num_value_heads:
            raise ValueError("GDN key and value head counts must match")
        if self.gdn_key_dim != self.gdn_value_dim:
            raise ValueError("GDN key and value head dimensions must match")
        if self.gdn_num_key_heads * self.gdn_key_dim != self.d_model:
            raise ValueError("GDN key/value width must equal d_model")

        if not isinstance(self.layer_pattern, tuple) or not self.layer_pattern:
            raise ValueError("layer_pattern must be a non-empty tuple")
        normalized_pattern: list[LayerKind] = []
        for kind in self.layer_pattern:
            if not isinstance(kind, str) or kind.lower() not in {"gdn", "gdn-2", "mha"}:
                raise ValueError("layer_pattern entries must be 'gdn' or 'mha'")
            normalized_pattern.append("mha" if kind.lower() == "mha" else "gdn")
        if tuple(normalized_pattern) != _DEFAULT_LAYER_PATTERN:
            raise ValueError("layer_pattern must be the frozen repeating ('gdn', 'gdn', 'gdn', 'mha') pattern")
        if self.n_layers % len(normalized_pattern):
            raise ValueError("n_layers must be divisible by the layer_pattern length")
        # Keep the frozen dataclass immutable while canonicalizing the documented GDN-2 spelling.
        object.__setattr__(self, "layer_pattern", tuple(normalized_pattern))

        if isinstance(self.rms_norm_eps, bool) or not isinstance(self.rms_norm_eps, (int, float)):
            raise ValueError("rms_norm_eps must be a positive finite number")
        if self.rms_norm_eps <= 0 or not math.isfinite(self.rms_norm_eps):
            raise ValueError("rms_norm_eps must be a positive finite number")
        if isinstance(self.rope_base, bool) or not isinstance(self.rope_base, (int, float)):
            raise ValueError("rope_base must be greater than 1 and finite")
        if self.rope_base <= 1 or not math.isfinite(self.rope_base):
            raise ValueError("rope_base must be greater than 1 and finite")
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)):
            raise ValueError("the frozen initial model requires dropout to be exactly 0")
        if self.dropout != 0 or not math.isfinite(self.dropout):
            raise ValueError("the frozen initial model requires dropout to be exactly 0")
        if self.attention_window is not None:
            if (
                isinstance(self.attention_window, bool)
                or not isinstance(self.attention_window, int)
                or self.attention_window <= 0
                or self.attention_window > self.max_seq_len
            ):
                raise ValueError("attention_window must be a positive integer no larger than max_seq_len")

    @property
    def layer_kinds(self) -> tuple[LayerKind, ...]:
        """Return the configured compact pattern expanded to every layer."""

        return self.layer_pattern * (self.n_layers // len(self.layer_pattern))

    @classmethod
    def smoke(cls, **overrides: object) -> ModelConfig:
        """Return the frozen approximately-20M smoke geometry."""

        values: dict[str, object] = {
            "d_model": 256,
            "n_layers": 8,
            "d_ff": 704,
            "n_heads": 4,
            "head_dim": 64,
            "gdn_num_key_heads": 4,
            "gdn_num_value_heads": 4,
            "gdn_key_dim": 64,
            "gdn_value_dim": 64,
        }
        values.update(overrides)
        return cls(**values)  # type: ignore[arg-type]

    @classmethod
    def substantive(cls, **overrides: object) -> ModelConfig:
        """Return the frozen first approximately-100M geometry."""

        values: dict[str, object] = {}
        values.update(overrides)
        return cls(**values)  # type: ignore[arg-type]


def smoke_config(**overrides: object) -> ModelConfig:
    """Functional spelling of :meth:`ModelConfig.smoke`."""

    return ModelConfig.smoke(**overrides)


def substantive_config(**overrides: object) -> ModelConfig:
    """Functional spelling of :meth:`ModelConfig.substantive`."""

    return ModelConfig.substantive(**overrides)
