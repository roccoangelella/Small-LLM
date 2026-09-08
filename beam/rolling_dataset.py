"""Beam bindings for the shared rolling-dataset orchestration."""
from __future__ import annotations

from pathlib import Path

import runtime as base_runtime
from profiles import canonical_run_id, resolve_presets
from providers import rolling_dataset as shared


def hf_dataset_bucket_id() -> str:
    return shared.hf_dataset_bucket_id(base_runtime=base_runtime)


def next_unconsumed_block(*, training_run_id: str, run_root: Path) -> dict[str, object]:
    return shared.next_unconsumed_block(
        training_run_id=training_run_id,
        run_root=run_root,
        base_runtime=base_runtime,
    )


def stage_for_h100(
    *,
    model: str,
    tokens: str,
    cache_root: Path,
    run_root: Path,
) -> dict[str, object]:
    return shared.stage_for_h100(
        model=model,
        tokens=tokens,
        cache_root=cache_root,
        run_root=run_root,
        base_runtime=base_runtime,
        resolve_presets=resolve_presets,
        canonical_run_id=canonical_run_id,
        provider_name="beam",
    )


def run_staged_training(
    *,
    model: str,
    tokens: str,
    source_commit: str,
    dataset_dir: str,
    max_steps_this_session: int,
    microbatch_size: int,
    precision: str,
    repo_root: Path,
    run_root: Path,
    cache_root: Path,
    run_volume: object,
    cache_volume: object,
) -> dict[str, object]:
    return shared.run_staged_training(
        model=model,
        tokens=tokens,
        source_commit=source_commit,
        dataset_dir=dataset_dir,
        max_steps_this_session=max_steps_this_session,
        microbatch_size=microbatch_size,
        precision=precision,
        repo_root=repo_root,
        run_root=run_root,
        cache_root=cache_root,
        run_volume=run_volume,
        cache_volume=cache_volume,
        base_runtime=base_runtime,
        resolve_presets=resolve_presets,
        canonical_run_id=canonical_run_id,
        provider_name="beam",
    )


__all__ = [
    "hf_dataset_bucket_id",
    "next_unconsumed_block",
    "run_staged_training",
    "stage_for_h100",
]
