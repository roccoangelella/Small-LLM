"""Numerically adaptive GDN-2 chunkwise backend for assembled training models."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .config import ModelConfig
from .gdn2 import (
    GDN2Backend,
    GatedDeltaNet2,
    _check_recurrence_inputs,
    gdn2_chunkwise_reference,
)

# Dense chunk products materialize both causal and discarded anti-causal
# entries before triangular masking. Keeping every cumulative-decay span below
# 60 leaves substantial FP32 exponent and reduction headroom.
_MAX_LOG_DECAY_SPAN = 60.0
_NONFINITE_CHUNK_MESSAGE = "chunkwise GDN-2 produced non-finite values"


def _log_decay_span(log_decay: Tensor, start: int, end: int) -> float:
    cumulative = torch.cumsum(log_decay[:, start:end].float().transpose(1, 2), dim=2)
    if not bool(torch.isfinite(cumulative).all()):
        return math.inf
    span = (cumulative.amax(dim=2) - cumulative.amin(dim=2)).amax()
    return float(span.detach().item())


class AdaptiveChunkwiseGDN2Backend:
    """Use the largest safe subchunks up to the configured chunk size.

    The correctness-first PyTorch chunk kernel factorizes pairwise decays into
    reciprocal exponentials. Strong but valid decay can overflow those
    intermediates even when the recurrent equations remain finite. This
    wrapper keeps ordinary chunks unchanged and bisects only numerically unsafe
    chunks. If a finite-span chunk still produces a non-finite result, it is
    retried at a smaller size down to the exact one-token recurrence.
    """

    def __init__(self, chunk_size: int = 64) -> None:
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError(f"chunk_size must be a positive integer, got {chunk_size!r}")
        self.chunk_size = chunk_size

    def __call__(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        log_decay: Tensor,
        erase_gate: Tensor,
        write_gate: Tensor,
        initial_state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        _, sequence, _, _ = _check_recurrence_inputs(
            q, k, v, log_decay, erase_gate, write_gate, initial_state
        )
        if sequence == 0:
            return gdn2_chunkwise_reference(
                q,
                k,
                v,
                log_decay,
                erase_gate,
                write_gate,
                initial_state,
                chunk_size=self.chunk_size,
            )

        outputs: list[Tensor] = []
        state = initial_state
        start = 0
        while start < sequence:
            end = min(start + self.chunk_size, sequence)
            while end - start > 1 and _log_decay_span(
                log_decay, start, end
            ) > _MAX_LOG_DECAY_SPAN:
                end = start + max(1, (end - start) // 2)

            while True:
                try:
                    output, candidate_state = gdn2_chunkwise_reference(
                        q[:, start:end],
                        k[:, start:end],
                        v[:, start:end],
                        log_decay[:, start:end],
                        erase_gate[:, start:end],
                        write_gate[:, start:end],
                        state,
                        chunk_size=end - start,
                    )
                except ValueError as error:
                    if _NONFINITE_CHUNK_MESSAGE not in str(error):
                        raise
                    current_size = end - start
                    if current_size == 1:
                        span = _log_decay_span(log_decay, start, end)
                        raise ValueError(
                            "adaptive chunkwise GDN-2 remained non-finite at "
                            f"token {start} with chunk_size=1 and log_decay_span={span:.6g}"
                        ) from error
                    end = start + max(1, current_size // 2)
                    continue

                outputs.append(output)
                state = candidate_state
                start = end
                break

        if state is None:
            raise RuntimeError("non-empty adaptive GDN-2 execution produced no final state")
        return torch.cat(outputs, dim=1), state


class StableGatedDeltaNet2(GatedDeltaNet2):
    """GDN-2 layer whose default training backend adapts unsafe chunks."""

    def __init__(self, config: ModelConfig, backend: GDN2Backend | None = None) -> None:
        super().__init__(
            config,
            backend=(
                backend
                if backend is not None
                else AdaptiveChunkwiseGDN2Backend(config.gdn_chunk_size)
            ),
        )


__all__ = ["AdaptiveChunkwiseGDN2Backend", "StableGatedDeltaNet2"]
