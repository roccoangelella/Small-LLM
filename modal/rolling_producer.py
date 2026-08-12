"""Modal-only orchestration for the concurrent incremental dataset producer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import runtime as base_runtime
from profiles import resolve_presets

APPROVED_WEIGHTS_SHA256 = "76e82e22760adcac59c7294fe9bac11358f5a8b7a26035aae64c3f2e6fa1acb7"


def _dataset_bucket_id() -> str:
    import os

    explicit = os.environ.get("SMALL_LLM_HF_DATASET_BUCKET_ID", "").strip()
    if explicit:
        return explicit
    return f"{base_runtime._hf_model_repo_id()}-datasets"


def produce_incremental_dataset(
    *,
    model: str,
    tokens: str,
    repo_root: Path,
    producer_root: Path,
    commit_cache_volume: Callable[[], object],
) -> dict[str, object]:
    """Build/publish READY shards with Volume commits on exact durability boundaries."""

    from dataset.incremental_frontier import SHARD_FRONTIER_FILENAME
    from dataset.production.cli import main as production_main
    from dataset.qualification import get_profile, production_arguments
    from dataset.src.hf_bucket_shards import HuggingFaceBucketShardStore
    from dataset.src.remote import sha256_path

    _, token_preset = resolve_presets(model, tokens)
    profile = get_profile(token_preset.dataset_profile)
    if not profile.incremental_frontier or profile.run_id is None:
        raise RuntimeError("concurrent producer was requested for a non-incremental dataset profile")

    bucket_id = _dataset_bucket_id()
    store = HuggingFaceBucketShardStore(
        bucket_id,
        token=base_runtime._hf_token(),
        private=True,
        create_bucket=True,
    )
    ready = store._read_json(store.object_key(profile.run_id, "ready.json"))
    frontier = store._read_json(store.object_key(profile.run_id, SHARD_FRONTIER_FILENAME))
    if (
        isinstance(ready, dict)
        and ready.get("target_reached") is True
        and isinstance(frontier, dict)
        and frontier.get("producer_complete") is True
    ):
        return {
            "status": "already_complete_remote",
            "dataset_run_id": profile.run_id,
            "dataset_bucket_id": bucket_id,
            "producer_complete": True,
        }

    weights = repo_root / "dataset" / "climbmix_code_free_weights.json"
    if not weights.is_file() or sha256_path(weights) != APPROVED_WEIGHTS_SHA256:
        raise RuntimeError("vendored ClimbMix production weights are missing or have the wrong SHA-256")
    output = producer_root / profile.run_id
    output.mkdir(parents=True, exist_ok=True)
    resume = (output / "work_plan.json").is_file()

    producer_args = [
        "--weights-file",
        str(weights),
        "--output-dir",
        str(output),
    ]
    if resume:
        producer_args.append("--resume")
    code = production_main(
        production_arguments(profile, producer_args),
        # The dataset builder invokes this only after atomically writing its
        # progress cursor and before making newly uploaded shards READY.
        durable_progress_hook=commit_cache_volume,
    )
    if code:
        raise RuntimeError(f"incremental dataset producer exited with status {code}")
    # Persist terminal manifest/frontier-adjacent local state too. READY safety
    # does not depend on this final commit, but future same-workspace inspection does.
    commit_cache_volume()

    final_frontier = store._read_json(store.object_key(profile.run_id, SHARD_FRONTIER_FILENAME))
    if not isinstance(final_frontier, dict) or final_frontier.get("producer_complete") is not True:
        raise RuntimeError("incremental producer returned without a completed remote frontier")
    result = {
        "status": "complete",
        "dataset_run_id": profile.run_id,
        "dataset_bucket_id": bucket_id,
        "producer_complete": True,
        "planned_train_blocks": final_frontier.get("planned_train_blocks"),
        "last_ready_train_block_id": final_frontier.get("last_ready_train_block_id"),
        "local_producer_dir": str(output),
    }
    print(json.dumps({"modal_cpu_dataset_producer": result}, sort_keys=True), flush=True)
    return result


__all__ = ["APPROVED_WEIGHTS_SHA256", "produce_incremental_dataset"]
