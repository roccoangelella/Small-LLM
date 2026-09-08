"""Modal bindings for the shared incremental dataset producer."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import runtime as base_runtime
from profiles import resolve_presets
from providers import rolling_producer as shared

APPROVED_WEIGHTS_SHA256 = shared.APPROVED_WEIGHTS_SHA256


def produce_incremental_dataset(
    *,
    model: str,
    tokens: str,
    repo_root: Path,
    producer_root: Path,
    commit_cache_volume: Callable[[], object],
) -> dict[str, object]:
    return shared.produce_incremental_dataset(
        model=model,
        tokens=tokens,
        repo_root=repo_root,
        producer_root=producer_root,
        commit_cache_volume=commit_cache_volume,
        base_runtime=base_runtime,
        resolve_presets=resolve_presets,
        provider_name="modal",
    )


__all__ = ["APPROVED_WEIGHTS_SHA256", "produce_incremental_dataset"]
