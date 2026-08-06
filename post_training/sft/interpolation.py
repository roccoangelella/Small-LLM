"""Configurable base/SFT parameter interpolation for retention experiments."""

from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch
from torch import Tensor


def interpolate_state_dicts(
    base_state: Mapping[str, Tensor],
    sft_state: Mapping[str, Tensor],
    *,
    alpha: float = 1.0,
) -> OrderedDict[str, Tensor]:
    """Return ``base + alpha * (sft - base)`` without mutating either input."""

    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError("alpha must be a finite number")
    alpha = float(alpha)
    if not torch.isfinite(torch.tensor(alpha)):
        raise ValueError("alpha must be finite")
    if set(base_state) != set(sft_state):
        missing = sorted(set(base_state) - set(sft_state))
        extra = sorted(set(sft_state) - set(base_state))
        raise ValueError(f"state dictionaries differ; missing={missing}, extra={extra}")

    result: OrderedDict[str, Tensor] = OrderedDict()
    for name in base_state:
        base = base_state[name]
        tuned = sft_state[name]
        if base.shape != tuned.shape or base.dtype != tuned.dtype:
            raise ValueError(f"state tensor geometry differs for {name!r}")
        if not (base.is_floating_point() or base.is_complex()):
            if alpha != 1.0 and not torch.equal(base, tuned):
                raise ValueError(
                    f"non-floating tensor {name!r} differs and cannot be interpolated"
                )
            result[name] = tuned.detach().clone() if alpha == 1.0 else base.detach().clone()
            continue
        work_dtype = torch.float32 if base.dtype in {torch.float16, torch.bfloat16} else base.dtype
        merged = base.detach().to(work_dtype).lerp(
            tuned.detach().to(work_dtype),
            alpha,
        )
        result[name] = merged.to(dtype=base.dtype)
    return result


__all__ = ["interpolate_state_dicts"]
