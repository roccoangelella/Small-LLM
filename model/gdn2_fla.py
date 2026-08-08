"""Flash Linear Attention execution backend for the existing Small-LLM GDN-2 layer.

This module deliberately changes only how the recurrence is evaluated.  The
Small-LLM projections, learned parameters, checkpoint keys, decay semantics,
and layer assembly remain owned by :mod:`model.gdn2`.
"""

from __future__ import annotations

import math
from typing import Any, Callable

from torch import Tensor

from .gdn2 import (
    GDN2Backend,
    _check_recurrence_inputs,
    _validate_backend_result,
)

FLA_CORE_VERSION = "0.5.1"
FLA_GDN2_CHUNK_SIZE = 64


def _load_chunk_gdn2() -> Callable[..., tuple[Tensor, Tensor | None]]:
    """Import the qualified FLA GDN-2 operator lazily.

    FLA is optional for CPU/reference use.  A CUDA training path that selects
    this backend must have the exact qualified core package installed rather
    than silently dropping back to the slow correctness implementation.
    """

    try:
        from fla.ops.gdn2 import chunk_gdn2
    except Exception as error:  # pragma: no cover - exercised on CUDA/Kaggle
        raise ImportError(
            "CUDA GDN-2 requires the qualified FLA core backend. Install "
            f"`fla-core=={FLA_CORE_VERSION}` (Kaggle qualification used "
            "`pip install --no-deps fla-core==0.5.1`)."
        ) from error
    return chunk_gdn2


class FLAGDN2Backend:
    """Evaluate the existing GDN-2 recurrence with FLA's Triton training kernel."""

    def __init__(self, *, chunk_size: int = FLA_GDN2_CHUNK_SIZE, disable_recompute: bool = False) -> None:
        if chunk_size != FLA_GDN2_CHUNK_SIZE:
            raise ValueError(
                f"FLA GDN-2 requires chunk_size={FLA_GDN2_CHUNK_SIZE}, got {chunk_size}"
            )
        self.chunk_size = chunk_size
        self.disable_recompute = bool(disable_recompute)

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
        if q.device.type != "cuda":
            raise NotImplementedError("FLA GDN-2 is selected only for CUDA execution")
        if sequence == 0:
            raise ValueError("FLA GDN-2 requires a non-empty sequence")

        chunk_gdn2 = _load_chunk_gdn2()
        output, final_state = chunk_gdn2(
            q=q,
            k=k,
            v=v,
            g=log_decay,
            b=erase_gate,
            w=write_gate,
            scale=1.0 / math.sqrt(q.shape[-1]),
            initial_state=initial_state,
            output_final_state=True,
            # Small-LLM already performs these transformations in its layer.
            use_qk_l2norm_in_kernel=False,
            use_gate_in_kernel=False,
            safe_gate=False,
            chunk_size=self.chunk_size,
            disable_recompute=self.disable_recompute,
        )
        if final_state is None:
            raise RuntimeError("FLA GDN-2 did not return the requested final recurrent state")
        return _validate_backend_result((output, final_state), q, v, initial_state)


class FLAPreferredGDN2Backend:
    """Use qualified FLA on the supported CUDA path and the reference fallback elsewhere.

    Selection happens at call time because Small-LLM models are commonly built
    on CPU and moved to CUDA afterwards.  On the active 64-token CUDA training
    geometry FLA is mandatory: an import/kernel failure is surfaced instead of
    silently reintroducing the adaptive backend's pathological runtime.
    """

    def __init__(
        self,
        *,
        chunk_size: int = FLA_GDN2_CHUNK_SIZE,
        fallback_backend: GDN2Backend,
        disable_recompute: bool = False,
    ) -> None:
        self.chunk_size = chunk_size
        self.fallback_backend = fallback_backend
        self.fla_backend = (
            FLAGDN2Backend(chunk_size=chunk_size, disable_recompute=disable_recompute)
            if chunk_size == FLA_GDN2_CHUNK_SIZE
            else None
        )

    def uses_fla_for(self, q: Tensor) -> bool:
        return q.device.type == "cuda" and self.fla_backend is not None

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
        if self.uses_fla_for(q):
            assert self.fla_backend is not None
            return self.fla_backend(
                q,
                k,
                v,
                log_decay,
                erase_gate,
                write_gate,
                initial_state,
            )
        return self.fallback_backend(
            q,
            k,
            v,
            log_decay,
            erase_gate,
            write_gate,
            initial_state,
        )


__all__ = [
    "FLA_CORE_VERSION",
    "FLA_GDN2_CHUNK_SIZE",
    "FLAGDN2Backend",
    "FLAPreferredGDN2Backend",
]
