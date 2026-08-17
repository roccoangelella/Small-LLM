"""Target-token mixture helpers for reasoning SFT plus instruction retention."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

REASONING_SHARE = 0.90
RETENTION_SHARE = 0.10
DEFAULT_REASONING_SOURCE = "r0-reasoning"


def _normalized_retention_shares(shares: Mapping[str, float]) -> dict[str, float]:
    if not shares:
        raise ValueError("retention_source_shares cannot be empty")
    normalized: dict[str, float] = {}
    for source, value in shares.items():
        if not isinstance(source, str) or not source.strip():
            raise ValueError("retention source names must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"retention share for {source!r} must be numeric")
        share = float(value)
        if not math.isfinite(share) or share <= 0:
            raise ValueError(f"retention share for {source!r} must be positive and finite")
        normalized[source.strip()] = share
    total = sum(normalized.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("retention_source_shares must sum to one")
    return normalized


def build_rsft_source_shares(
    retention_source_shares: Mapping[str, float],
    *,
    reasoning_source: str = DEFAULT_REASONING_SOURCE,
) -> dict[str, float]:
    """Scale an arbitrary retention sub-mixture into the frozen 90/10 top-level mix."""

    if not isinstance(reasoning_source, str) or not reasoning_source.strip():
        raise ValueError("reasoning_source must be a non-empty string")
    reasoning_source = reasoning_source.strip()
    retention = _normalized_retention_shares(retention_source_shares)
    if reasoning_source in retention:
        raise ValueError("reasoning_source cannot also be a retention source")
    result = {reasoning_source: REASONING_SHARE}
    result.update({source: RETENTION_SHARE * share for source, share in retention.items()})
    return result


def build_target_token_mixer(
    *,
    reasoning_records: Iterable[Any],
    retention_sources: Mapping[str, Iterable[Any]],
    retention_source_shares: Mapping[str, float],
    target_loss_tokens: int,
    seed: int,
    reasoning_source: str = DEFAULT_REASONING_SOURCE,
) -> Any:
    """Reuse the existing SFT mixer so the 10% retention share is loss-token based."""

    if set(retention_sources) != set(retention_source_shares):
        raise ValueError("retention_sources and retention_source_shares must have identical keys")
    from post_training.sft.mixture import TargetTokenMixer

    sources = {reasoning_source: reasoning_records, **dict(retention_sources)}
    shares = build_rsft_source_shares(
        retention_source_shares,
        reasoning_source=reasoning_source,
    )
    return TargetTokenMixer(
        sources,
        shares,
        seed=seed,
        target_loss_tokens=target_loss_tokens,
    )


__all__ = [
    "DEFAULT_REASONING_SOURCE",
    "REASONING_SHARE",
    "RETENTION_SHARE",
    "build_rsft_source_shares",
    "build_target_token_mixer",
]
