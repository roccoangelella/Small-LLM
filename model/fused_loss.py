"""Chunked tied-embedding cross-entropy that never materializes full logits.

The training loss used to be ``F.cross_entropy(model(ids).reshape(-1, V), labels)``
over logits of shape [tokens, V]. For 131,072 targets and V=50,304 that is
~13 GB of BF16 logits per update (2 GB at V=8,000), kept alive for backward.
This module computes the same summed cross-entropy in token chunks and
recomputes each chunk's logits during backward (non-reentrant activation
checkpointing), so the live logits buffer is bounded by ``chunk_tokens`` × V.

Contract: the value equals the unchunked sum up to floating-point summation
order; gradients with respect to the hidden states and the tied weight are the
same up to the same rounding; ``ignore_index`` positions contribute nothing,
exactly as in the unchunked call. Extra cost: one additional output-projection
matmul per chunk during backward (recompute).
"""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

DEFAULT_CHUNK_TOKENS = 4096


def _chunk_cross_entropy_sum(
    hidden: Tensor, weight: Tensor, labels: Tensor, ignore_index: int
) -> Tensor:
    logits = F.linear(hidden, weight)
    return F.cross_entropy(logits.float(), labels, reduction="sum", ignore_index=ignore_index)


def chunked_cross_entropy_sum(
    hidden: Tensor,
    weight: Tensor,
    labels: Tensor,
    *,
    semantic_vocab_size: int,
    ignore_index: int,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
) -> Tensor:
    """Summed cross-entropy of ``hidden @ weight[:semantic].T`` against labels.

    ``hidden``: [..., d_model]; ``labels``: matching leading shape; ``weight``:
    the tied embedding matrix [padded_vocab, d_model], of which only the first
    ``semantic_vocab_size`` rows are logits (padded rows are never scored).
    """

    if hidden.shape[:-1] != labels.shape:
        raise RuntimeError("hidden states do not match training labels")
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    if weight.shape[0] < semantic_vocab_size:
        raise ValueError("tied weight has fewer rows than the semantic vocabulary")
    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    flat_labels = labels.reshape(-1)
    semantic_weight = weight[:semantic_vocab_size]
    total = torch.zeros((), dtype=torch.float32, device=hidden.device)
    tokens = flat_hidden.shape[0]
    for start in range(0, tokens, chunk_tokens):
        stop = min(tokens, start + chunk_tokens)
        chunk = checkpoint(
            _chunk_cross_entropy_sum,
            flat_hidden[start:stop],
            semantic_weight,
            flat_labels[start:stop],
            ignore_index,
            use_reentrant=False,
        )
        total = total + chunk
    return total


__all__ = ["DEFAULT_CHUNK_TOKENS", "chunked_cross_entropy_sum"]
