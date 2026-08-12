"""Modal-only orchestration for the HF rolling dataset transport.

All dataset transport semantics live in :mod:`dataset`.  This module only binds
those provider-neutral primitives to Modal checkpoint/cache Volumes and enforces
the allocation order: resolve the next checkpoint block on CPU, download and
verify that dataset shard on CPU, commit it to the cache Volume, and only then
permit H100 dispatch.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import runtime as base_runtime
from profiles import canonical_run_id, resolve_presets


def hf_dataset_bucket_id() -> str:
    explicit = os.environ.get("SMALL_LLM_HF_DATASET_BUCKET_ID", "").strip()
    if explicit:
        return explicit
    return f"{base_runtime._hf_model_repo_id()}-datasets"


def _pointer_step(pointer: object) -> int:
    if pointer is None:
        return 0
    if not isinstance(pointer, Mapping):
        raise RuntimeError("Hugging Face checkpoint latest pointer is not a JSON object")
    checkpoint_id = pointer.get("checkpoint_id")
    if not isinstance(checkpoint_id, str):
        raise RuntimeError("Hugging Face checkpoint latest pointer has no checkpoint_id")
    match = base_runtime._CHECKPOINT_ID.fullmatch(checkpoint_id)
    if match is None:
        raise RuntimeError("Hugging Face checkpoint latest pointer has an invalid checkpoint_id")
    return int(match.group(1))


def next_unconsumed_block(*, training_run_id: str, run_root: Path) -> dict[str, object]:
    """Resolve the exact next block from the newest local or HF durable checkpoint."""

    local_id, local_step = base_runtime._latest_checkpoint(
        run_root / training_run_id / "checkpoints"
    )
    remote_store = base_runtime._hf_bucket_store()
    pointer = remote_store.read_json(f"run/{training_run_id}/latest.json")
    remote_step = _pointer_step(pointer)
    step = max(local_step, remote_step)
    source = "fresh"
    if step == local_step and local_step > remote_step:
        source = "modal_volume"
    elif step == remote_step and remote_step > 0:
        source = "hf_bucket"
    elif step > 0:
        source = "modal_volume+hf_bucket"
    return {
        "completed_steps": step,
        "next_block_id": step,
        "checkpoint_source": source,
        "local_checkpoint_id": local_id,
        "local_step": local_step,
        "remote_step": remote_step,
    }


def stage_for_h100(
    *,
    model: str,
    tokens: str,
    cache_root: Path,
    run_root: Path,
) -> dict[str, object]:
    """Download+SHA256-verify the checkpoint-aligned shard before GPU allocation."""

    from dataset.qualification import get_profile
    from dataset.rolling_cache import stage_dataset_window
    from dataset.src.hf_bucket_shards import HuggingFaceBucketShardStore

    model_preset, token_preset = resolve_presets(model, tokens)
    if token_preset.dataset_transport != "hf_rolling_shards":
        raise RuntimeError("CPU rolling-dataset staging was requested for a non-rolling profile")
    profile = get_profile(token_preset.dataset_profile)
    if profile.run_id is None:
        raise RuntimeError("rolling dataset profile has no immutable run ID")
    training_run_id = canonical_run_id(model_preset, token_preset)
    cursor = next_unconsumed_block(training_run_id=training_run_id, run_root=run_root)
    bucket_id = hf_dataset_bucket_id()
    store = HuggingFaceBucketShardStore(
        bucket_id,
        token=base_runtime._hf_token(),
        private=True,
        create_bucket=False,
    )
    destination = cache_root / "datasets" / profile.run_id
    staged = stage_dataset_window(
        store=store,
        run_id=profile.run_id,
        destination=destination,
        start_block_id=int(cursor["next_block_id"]),
        train_shards=1,
    )
    result = {
        **staged,
        **cursor,
        "training_run_id": training_run_id,
        "dataset_profile": token_preset.dataset_profile,
        "dataset_run_id": profile.run_id,
        "dataset_bucket_id": bucket_id,
        "h100_dispatch_allowed": staged.get("status") == "ready",
    }
    print(json.dumps({"modal_cpu_dataset_stage": result}, sort_keys=True), flush=True)
    return result


def _preauthorize_existing_runtime_verification(
    *,
    run_dir: Path,
    dataset: Path,
    required_block: int,
    bucket_id: str,
    dataset_run_id: str,
) -> dict[str, object]:
    """Recheck CPU staging and write the marker consumed by the legacy Modal runtime.

    The dataset verifier hashes the checkpoint-aligned train shard and all
    validation shards again from the shared Modal cache Volume. No network fetch
    happens here; the expensive first shard download was completed by CPU before
    this H100 function was spawned.
    """

    from dataset.rolling_cache import verify_staged_dataset

    verification = verify_staged_dataset(
        destination=dataset,
        bucket_id=bucket_id,
        run_id=dataset_run_id,
        required_train_block=required_block,
    )
    identity = {
        "dataset": str(dataset),
        "manifest_sha256": base_runtime._sha256(dataset / "manifest.json"),
        # Match base runtime's local identity exactly so it does not attempt a
        # whole-dataset full-scan that is intentionally impossible here.
        "transport": "modal_volume",
    }
    base_runtime._write_json(
        run_dir / "dataset_verified.json",
        {
            "identity": identity,
            "datasets_inspected": [
                {
                    "root": str(dataset),
                    "transport": "hf_rolling_shards",
                    "cpu_staged": True,
                    "required_train_block": required_block,
                    "verification": verification,
                }
            ],
        },
    )
    return verification


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
    """Run the ordinary Modal trainer against the CPU-staged rolling dataset."""

    from dataset.qualification import get_profile

    model_preset, token_preset = resolve_presets(model, tokens)
    if token_preset.dataset_transport != "hf_rolling_shards":
        raise RuntimeError("rolling H100 runtime received a non-rolling token profile")
    profile = get_profile(token_preset.dataset_profile)
    if profile.run_id is None:
        raise RuntimeError("rolling dataset profile has no immutable run ID")
    training_run_id = canonical_run_id(model_preset, token_preset)
    dataset = Path(dataset_dir)
    if not dataset.is_absolute():
        raise RuntimeError("rolling Modal dataset directory must be an absolute CPU-staged path")
    cache_resolved = cache_root.resolve(strict=True)
    dataset_resolved = dataset.resolve(strict=True)
    try:
        dataset_resolved.relative_to(cache_resolved)
    except ValueError as error:
        raise RuntimeError("rolling Modal dataset must live inside the cache Volume") from error

    bucket_id = hf_dataset_bucket_id()
    staged = json.loads((dataset_resolved / "rolling_cache_stage.json").read_text(encoding="utf-8"))
    if not isinstance(staged, Mapping):
        raise RuntimeError("rolling dataset staging marker is invalid")
    required_block = staged.get("start_block_id")
    if isinstance(required_block, bool) or not isinstance(required_block, int) or required_block < 0:
        raise RuntimeError("rolling dataset staging marker has an invalid start block")
    if staged.get("training_complete") is True:
        raise RuntimeError("H100 was allocated even though CPU staging marked training complete")

    # Fail rather than silently downloading a different resume shard on the H100
    # if the checkpoint pointer changed between CPU staging and GPU dispatch.
    current = next_unconsumed_block(training_run_id=training_run_id, run_root=run_root)
    if int(current["next_block_id"]) != required_block:
        raise RuntimeError(
            "checkpoint advanced after CPU dataset staging; restage on CPU before allocating H100"
        )
    run_dir = run_root / training_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    verification = _preauthorize_existing_runtime_verification(
        run_dir=run_dir,
        dataset=dataset_resolved,
        required_block=required_block,
        bucket_id=bucket_id,
        dataset_run_id=profile.run_id,
    )

    os.environ["SMALL_LLM_MODAL_ROLLING_DATASET"] = "1"
    os.environ["SMALL_LLM_DATASET_SHARD_BUCKET"] = bucket_id
    os.environ["SMALL_LLM_DATASET_SHARD_RUN_ID"] = profile.run_id
    os.environ["SMALL_LLM_DATASET_SHARD_PREFETCH"] = "1"

    result = base_runtime.run_training(
        model=model,
        tokens=tokens,
        source_commit=source_commit,
        dataset_dir=str(dataset_resolved),
        max_steps_this_session=max_steps_this_session,
        microbatch_size=microbatch_size,
        precision=precision,
        repo_root=repo_root,
        data_root=cache_root,
        run_root=run_root,
        cache_root=cache_root,
        run_volume=run_volume,
        cache_volume=cache_volume,
    )
    runtime_path = run_dir / "modal_runtime.json"
    if runtime_path.is_file():
        contract = base_runtime._json(runtime_path)
        contract.update(
            dataset_transport="hf_rolling_shards",
            dataset_bucket_id=bucket_id,
            dataset_prefetch_shards=1,
            cpu_staged_required_block=required_block,
        )
        base_runtime._write_json(runtime_path, contract)
        getattr(run_volume, "commit")()
    result = dict(result)
    result["dataset_transport"] = {
        "kind": "hf_rolling_shards",
        "bucket": bucket_id,
        "prefetch_shards": 1,
        "cpu_staged_required_block": required_block,
        "cpu_stage_verification": verification,
    }
    return result


__all__ = [
    "hf_dataset_bucket_id",
    "next_unconsumed_block",
    "run_staged_training",
    "stage_for_h100",
]
