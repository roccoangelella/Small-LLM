"""Flash Linear Attention execution backend for the existing Small-LLM GDN-2 layer.

This module deliberately changes only how the recurrence is evaluated. The
Small-LLM projections, learned parameters, checkpoint keys, decay semantics,
and layer assembly remain owned by :mod:`model.gdn2`.
"""

from __future__ import annotations

import math
from typing import Callable

from torch import Tensor

from .gdn2 import (
    GDN2Backend,
    _check_recurrence_inputs,
    _validate_backend_result,
)

FLA_CORE_VERSION = "0.5.1"
FLA_GDN2_CHUNK_SIZE = 64


def _load_chunk_gdn2() -> Callable[..., tuple[Tensor, Tensor | None]]:
    """Import the qualified FLA GDN-2 operator lazily."""

    try:
        from fla.ops.gdn2 import chunk_gdn2
    except Exception as error:  # pragma: no cover - exercised on CUDA/Kaggle
        raise ImportError(
            "CUDA GDN-2 requires the qualified FLA core backend. Install "
            f"`fla-core=={FLA_CORE_VERSION}` (Kaggle qualification used "
            "`pip install --no-deps fla-core==0.5.1`)."
        ) from error
    return chunk_gdn2


def _fla_compute_tensors(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    erase_gate: Tensor,
    write_gate: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Give FLA one common compute dtype for its Tensor-Core dot operands.

    Small-LLM training keeps parameters in FP32 and uses CUDA autocast.  Under
    that real trainer path, q/k can emerge from L2 normalization in FP32 while
    v and the write-side projection are FP16.  FLA's WY recompute allocates its
    solved matrix with k.dtype and later dots it with a v/write block; Triton
    requires those dot operands to have the same dtype.

    The standalone qualification used FP16 q/k/v/erase/write and FP32 log
    decay/state.  Canonicalizing the five ordinary compute tensors to v.dtype
    reproduces that qualified mixed-precision contract.  log_decay and the
    recurrent state deliberately remain FP32 outside this helper.
    """

    dtype = v.dtype
    return tuple(
        tensor if tensor.dtype == dtype else tensor.to(dtype=dtype)
        for tensor in (q, k, v, erase_gate, write_gate)
    )  # type: ignore[return-value]


class FLAGDN2Backend:
    """Evaluate the existing GDN-2 recurrence with FLA's Triton training kernel."""

    def __init__(self, *, disable_recompute: bool = False) -> None:
        self.chunk_size = FLA_GDN2_CHUNK_SIZE
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

        q_fla, k_fla, v_fla, erase_fla, write_fla = _fla_compute_tensors(
            q, k, v, erase_gate, write_gate
        )
        chunk_gdn2 = _load_chunk_gdn2()
        output, final_state = chunk_gdn2(
            q=q_fla,
            k=k_fla,
            v=v_fla,
            g=log_decay,
            b=erase_fla,
            w=write_fla,
            scale=1.0 / math.sqrt(q.shape[-1]),
            initial_state=initial_state,
            output_final_state=True,
            # Small-LLM already performs these transformations in its layer.
            use_qk_l2norm_in_kernel=False,
            use_gate_in_kernel=False,
            safe_gate=False,
            chunk_size=FLA_GDN2_CHUNK_SIZE,
            disable_recompute=self.disable_recompute,
        )
        if final_state is None:
            raise RuntimeError("FLA GDN-2 did not return the requested final recurrent state")
        # Preserve the Small-LLM backend contract seen by the rest of the layer;
        # the internal cast remains differentiable and gradients flow back to
        # the original autocast-produced tensors/FP32 parameters.
        if output.dtype != q.dtype:
            output = output.to(dtype=q.dtype)
        return _validate_backend_result((output, final_state), q, v, initial_state)


class FLAPreferredGDN2Backend:
    """Use qualified FLA on CUDA and the configured adaptive backend elsewhere.

    ``configured_chunk_size`` is intentionally preserved as checkpoint/model
    configuration because historical checkpoints record it. FLA's GDN-2
    kernel itself is fixed to 64-token blocks. Since chunk size changes only
    execution grouping and not the recurrence semantics, CUDA calls may use the
    fixed FLA-64 kernel even when a checkpoint was created with the historical
    adaptive ``gdn_chunk_size=32`` setting.

    This distinction is what makes an existing 500M checkpoint load strictly
    unchanged while still replacing the pathological CUDA calculator.
    """

    def __init__(
        self,
        *,
        chunk_size: int,
        fallback_backend: GDN2Backend,
        disable_recompute: bool = False,
    ) -> None:
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError(f"chunk_size must be a positive integer, got {chunk_size!r}")
        self.chunk_size = chunk_size
        self.fallback_backend = fallback_backend
        self.fla_backend = FLAGDN2Backend(disable_recompute=disable_recompute)

    def uses_fla_for(self, q: Tensor) -> bool:
        return q.device.type == "cuda"

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
