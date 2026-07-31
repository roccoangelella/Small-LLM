"""Readable PyTorch reference components for the frozen model contract."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    """RMS normalization with one learned scale and no learned bias."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim!r}")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError(f"eps must be positive, got {eps!r}")
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        variance = x.float().square().mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance + self.eps).to(dtype=x.dtype)
        return normalized * self.weight


class RotaryEmbedding(nn.Module):
    """Fixed full-head RoPE for tensors laid out as ``[B, T, H, D]``."""

    def __init__(self, head_dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        if isinstance(head_dim, bool) or not isinstance(head_dim, int) or head_dim <= 0:
            raise ValueError("head_dim must be a positive integer")
        if head_dim % 2:
            raise ValueError(f"RoPE head_dim must be even, got {head_dim}")
        if not math.isfinite(base) or base <= 1:
            raise ValueError("RoPE base must be greater than 1")
        self.head_dim = head_dim
        self.base = base

    def forward(self, x: Tensor, positions: Tensor | None = None) -> Tensor:
        if x.ndim != 4 or x.shape[-1] != self.head_dim:
            raise ValueError(f"expected [batch, sequence, heads, {self.head_dim}], got {tuple(x.shape)}")
        batch, sequence = x.shape[:2]
        if positions is None:
            positions = torch.arange(sequence, device=x.device)
        if positions.ndim == 1:
            if positions.shape[0] != sequence:
                raise ValueError("positions must have length equal to the sequence length")
            positions = positions.unsqueeze(0).expand(batch, -1)
        elif positions.ndim == 2:
            if positions.shape != (batch, sequence):
                raise ValueError("batched positions must have shape [batch, sequence]")
        else:
            raise ValueError("positions must have shape [sequence] or [batch, sequence]")
        if positions.device != x.device:
            positions = positions.to(device=x.device)
        if positions.is_floating_point() or positions.is_complex():
            raise ValueError("positions must be an integer tensor")

        inverse_frequency = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.head_dim, 2, device=x.device, dtype=torch.float32)
                / self.head_dim
            )
        )
        angles = positions.to(torch.float32).unsqueeze(-1) * inverse_frequency
        cos = angles.cos().to(dtype=x.dtype).unsqueeze(2)
        sin = angles.sin().to(dtype=x.dtype).unsqueeze(2)
        first, second = x[..., ::2], x[..., 1::2]
        return torch.stack((first * cos - second * sin, first * sin + second * cos), dim=-1).flatten(-2)


def apply_rope(
    x: Tensor,
    positions: Tensor | None = None,
    *,
    base: float = 10_000.0,
    seq_dim: int = -2,
) -> Tensor:
    """Apply the same fixed RoPE to a tensor with an explicit sequence axis."""

    if x.ndim != 4:
        raise ValueError("RoPE input must have shape [batch, heads, sequence, head_dim]")
    seq_dim = seq_dim if seq_dim >= 0 else x.ndim + seq_dim
    if seq_dim not in (1, 2):
        raise ValueError("seq_dim must identify the heads or sequence axis")
    moved = x.movedim(seq_dim, 1)
    rotated = RotaryEmbedding(x.shape[-1], base)(moved, positions)
    return rotated.movedim(1, seq_dim)


def apply_rope_qk(
    q: Tensor,
    k: Tensor,
    positions: Tensor | None = None,
    *,
    base: float = 10_000.0,
) -> tuple[Tensor, Tensor]:
    """Apply fixed RoPE to Q and K, leaving V unmodified by contract."""

    if q.shape != k.shape:
        raise ValueError("Q and K must have matching shapes")
    return apply_rope(q, positions, base=base), apply_rope(k, positions, base=base)


class SwiGLU(nn.Module):
    """Bias-free SiLU-gated feed-forward projection."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TiedEmbedding(nn.Module):
    """Tied token embedding and semantic-vocabulary LM projection."""

    def __init__(
        self,
        d_model: int | ModelConfig,
        semantic_vocab_size: int = 50_257,
        padded_vocab_size: int = 50_304,
    ) -> None:
        super().__init__()
        if isinstance(d_model, ModelConfig):
            config = d_model
            d_model = config.d_model
            semantic_vocab_size = config.semantic_vocab_size
            padded_vocab_size = config.padded_vocab_size
        if semantic_vocab_size <= 0 or padded_vocab_size < semantic_vocab_size:
            raise ValueError("padded vocabulary must contain the semantic vocabulary")
        if padded_vocab_size % 8:
            raise ValueError("padded_vocab_size must be divisible by 8")
        self.semantic_vocab_size = semantic_vocab_size
        self.padded_vocab_size = padded_vocab_size
        self.weight = nn.Parameter(torch.empty(padded_vocab_size, d_model))
        nn.init.normal_(self.weight)
        with torch.no_grad():
            self.weight[semantic_vocab_size:].zero_()

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("input_ids must be an integer tensor")
        if input_ids.numel() and (
            bool((input_ids < 0).any()) or bool((input_ids >= self.semantic_vocab_size).any())
        ):
            raise ValueError(
                f"input IDs must be in [0, {self.semantic_vocab_size}); padded rows are not semantic tokens"
            )
        return F.embedding(input_ids.to(dtype=torch.long), self.weight)

    def logits(self, hidden: Tensor) -> Tensor:
        aligned_logits = F.linear(hidden, self.weight)
        return aligned_logits[..., : self.semantic_vocab_size]

    def lm_head(self, hidden: Tensor) -> Tensor:
        return self.logits(hidden)


TiedTokenEmbedding = TiedEmbedding
TiedEmbeddingOutput = TiedEmbedding


class GatedMultiheadAttention(nn.Module):
    """Full-head causal MHA with QK normalization, RoPE, and an output gate."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.gate_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.rotary = RotaryEmbedding(config.head_dim, config.rope_base)

    def forward(self, x: Tensor, positions: Tensor | None = None) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.config.d_model:
            raise ValueError(f"expected input [batch, sequence, {self.config.d_model}], got {tuple(x.shape)}")
        batch, sequence, _ = x.shape
        if sequence > self.config.max_seq_len:
            raise ValueError("sequence length cannot exceed max_seq_len")

        q = self.q_proj(x).view(batch, sequence, self.config.n_heads, self.config.head_dim)
        k = self.k_proj(x).view(batch, sequence, self.config.n_heads, self.config.head_dim)
        v = self.v_proj(x).view(batch, sequence, self.config.n_heads, self.config.head_dim)
        q = self.rotary(self.q_norm(q), positions)
        k = self.rotary(self.k_norm(k), positions)

        # [B, T, H, D] -> [B, H, T, D] for the attention contraction.
        # Keep the score calculation in FP32.  This reference path favors
        # stable, obvious masking semantics over a fused attention kernel.
        scores = torch.einsum("bthd,bshd->bhts", q.float(), k.float()) / math.sqrt(
            self.config.head_dim
        )
        token_indices = torch.arange(sequence, device=x.device)
        allowed = token_indices[:, None] >= token_indices[None, :]
        if self.config.attention_window is not None:
            allowed = allowed & (
                token_indices[:, None] - token_indices[None, :] < self.config.attention_window
            )
        scores = scores.masked_fill(~allowed.view(1, 1, sequence, sequence), float("-inf"))
        attention = F.softmax(scores, dim=-1).to(dtype=v.dtype)
        mixed = torch.einsum("bhts,bshd->bthd", attention, v).reshape(
            batch, sequence, self.config.d_model
        )
        gated = mixed * torch.sigmoid(self.gate_proj(x))
        return self.out_proj(gated)


# Compatibility spelling for callers that used the earlier descriptive name.
CausalSelfAttention = GatedMultiheadAttention
CausalMHA = GatedMultiheadAttention
