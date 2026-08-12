"""Modal-only orchestration for the concurrent incremental dataset producer."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Callable

import runtime as base_runtime
from profiles import resolve_presets

APPROVED_WEIGHTS_SHA256 = "76e82e22760adcac59c7294fe9bac11358f5a8b7a26035aae64c3f2e6fa1acb7"
COMMIT_INTERVAL_SECONDS = 60.0


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
    """Build/publish READY shards while periodically committing Modal producer state."""

    from dataset.incremental_frontier import SHARD_FRONTIER_FILENAME
    from dataset.qualification import get_profile, main as dataset_main
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

    stop = threading.Event()
    commit_errors: list[BaseException] = []

    def commit_loop() -> None:
        while not stop.wait(COMMIT_INTERVAL_SECONDS):
            try:
                commit_cache_volume()
            except BaseException as error:  # noqa: BLE001 - surfaced at producer boundary
                commit_errors.append(error)
                stop.set()
                return

    thread = threading.Thread(target=commit_loop, name="modal-dataset-producer-volume-commit", daemon=True)
    thread.start()
    try:
        argv = [
            "build",
            "--profile",
            profile.key,
            "--weights-file",
            str(weights),
            "--output-dir",
            str(output),
        ]
        if resume:
            argv.append("--resume")
        code = dataset_main(argv)
        if code:
            raise RuntimeError(f"incremental dataset producer exited with status {code}")
        if commit_errors:
            raise RuntimeError(
                "Modal cache Volume commit failed during dataset production: "
                f"{type(commit_errors[0]).__name__}: {commit_errors[0]}"
            )
        commit_cache_volume()
    finally:
        stop.set()
        thread.join(timeout=5.0)

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
