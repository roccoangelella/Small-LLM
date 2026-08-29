"""Execution-only checkpoint topology rewrites for the 100M/10B deep-decay run.

The scientific checkpoint state is provider-neutral. Providers may differ only
in execution slicing and CUDA RNG topology:

- Kaggle 2xT4: microbatch 2, two CUDA RNG states;
- Beam single GPU: microbatch 4, one CUDA RNG state;
- Modal H100: microbatch 16, one CUDA RNG state.

A one-GPU -> two-GPU migration duplicates rank zero's exact CUDA RNG byte state
to both ranks. A two-GPU -> one-GPU migration projects rank zero. No optimizer,
model, scheduler, data-cursor, CPU RNG, or consumed-token state is modified.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

AUTHORIZED_MICROBATCHES = frozenset({2, 4, 16})
TARGET_CUDA_RNG_STATE_COUNTS = {2: 2, 4: 1, 16: 1}
SOURCE_CUDA_RNG_STATE_COUNTS = {
    2: frozenset({1, 2}),  # one-state legacy Kaggle checkpoints are canonicalized on load
    4: frozenset({1}),
    16: frozenset({1}),
}


def _clone_rng_state(value: object) -> object:
    clone = getattr(value, "clone", None)
    if callable(clone):
        return clone()
    return copy.deepcopy(value)


def validate_execution_state(
    state: Mapping[str, object],
    *,
    allowed_microbatches: frozenset[int] = AUTHORIZED_MICROBATCHES,
) -> tuple[int, int]:
    """Validate provider-varying fields and return (microbatch, CUDA RNG count)."""

    config = state.get("config")
    scheduler = state.get("scheduler")
    if not isinstance(config, Mapping) or not isinstance(scheduler, Mapping):
        raise RuntimeError("checkpoint lacks config/scheduler mappings")

    saved_microbatch = config.get("microbatch_size")
    if (
        isinstance(saved_microbatch, bool)
        or not isinstance(saved_microbatch, int)
        or saved_microbatch not in allowed_microbatches
    ):
        raise RuntimeError(
            f"checkpoint execution microbatch {saved_microbatch!r} is not an authorized migration source"
        )

    scheduler_config = scheduler.get("config")
    if not isinstance(scheduler_config, Mapping):
        raise RuntimeError("checkpoint scheduler lacks config mapping")
    if dict(scheduler_config) != dict(config):
        raise RuntimeError("checkpoint scheduler config disagrees with trainer config")

    cuda_rng_states = state.get("cuda_rng_states")
    if not isinstance(cuda_rng_states, list) or not cuda_rng_states:
        raise RuntimeError("checkpoint lacks CUDA RNG state list")
    allowed_counts = SOURCE_CUDA_RNG_STATE_COUNTS[saved_microbatch]
    if len(cuda_rng_states) not in allowed_counts:
        raise RuntimeError(
            "checkpoint CUDA RNG topology drifted: "
            f"microbatch={saved_microbatch!r}, states={len(cuda_rng_states)}"
        )
    return saved_microbatch, len(cuda_rng_states)


def execution_rewrite_needed(
    state: Mapping[str, object],
    *,
    target_microbatch: int,
) -> bool:
    """Return whether state needs execution canonicalization for the target lane."""

    if target_microbatch not in TARGET_CUDA_RNG_STATE_COUNTS:
        raise ValueError(f"unsupported target microbatch: {target_microbatch}")
    saved_microbatch, saved_rng_count = validate_execution_state(state)
    return (
        saved_microbatch != target_microbatch
        or saved_rng_count != TARGET_CUDA_RNG_STATE_COUNTS[target_microbatch]
    )


def rewrite_execution_state(
    state: Mapping[str, object],
    *,
    target_microbatch: int,
) -> tuple[dict[str, object], dict[str, Any]]:
    """Return a provider-canonical state plus explicit migration metadata."""

    if target_microbatch not in TARGET_CUDA_RNG_STATE_COUNTS:
        raise ValueError(f"unsupported target microbatch: {target_microbatch}")

    source_microbatch, source_rng_count = validate_execution_state(state)
    target_rng_count = TARGET_CUDA_RNG_STATE_COUNTS[target_microbatch]
    cuda_rng_states = state["cuda_rng_states"]
    assert isinstance(cuda_rng_states, list) and cuda_rng_states

    patched = dict(state)
    config = dict(state["config"])  # type: ignore[arg-type]
    scheduler = dict(state["scheduler"])  # type: ignore[arg-type]
    config["microbatch_size"] = target_microbatch
    scheduler["config"] = dict(config)
    patched["config"] = config
    patched["scheduler"] = scheduler

    if target_rng_count == 1:
        patched_rng_states = [_clone_rng_state(cuda_rng_states[0])]
        rng_policy = "project_rank0"
    elif source_microbatch == target_microbatch and source_rng_count == target_rng_count:
        patched_rng_states = [_clone_rng_state(value) for value in cuda_rng_states]
        rng_policy = "preserve_per_rank"
    else:
        patched_rng_states = [
            _clone_rng_state(cuda_rng_states[0]) for _ in range(target_rng_count)
        ]
        rng_policy = "duplicate_rank0"
    patched["cuda_rng_states"] = patched_rng_states

    metadata: dict[str, Any] = {
        "changed": (
            source_microbatch != target_microbatch
            or source_rng_count != target_rng_count
        ),
        "from_microbatch": source_microbatch,
        "to_microbatch": target_microbatch,
        "cuda_rng_states": {"from": source_rng_count, "to": target_rng_count},
        "cuda_rng_policy": rng_policy,
    }
    return patched, metadata


def validate_target_execution_state(
    state: Mapping[str, object],
    *,
    target_microbatch: int,
) -> None:
    """Require exactly the canonical execution topology for one provider lane."""

    saved_microbatch, saved_rng_count = validate_execution_state(
        state,
        allowed_microbatches=frozenset({target_microbatch}),
    )
    expected_rng_count = TARGET_CUDA_RNG_STATE_COUNTS[target_microbatch]
    if saved_microbatch != target_microbatch or saved_rng_count != expected_rng_count:
        raise RuntimeError(
            "checkpoint execution topology is not canonical for target lane: "
            f"microbatch={saved_microbatch}, cuda_rng_states={saved_rng_count}, "
            f"expected={target_microbatch}/{expected_rng_count}"
        )
