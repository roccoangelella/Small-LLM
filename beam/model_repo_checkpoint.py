"""Adapt the existing Modal runtime to use the HF model repository for checkpoints.

The core runtime remains frozen.  This adapter changes checkpoint transport only:

- live exact-resume checkpoints use the existing two-phase ``run/<run_id>/...``
  model-repository protocol with rolling latest-only cleanup;
- the final completed checkpoint is also published under the stable
  ``models/<run_id>/<checkpoint_id>`` namespace;
- the former HF Storage Bucket remains accepted only as a legacy restore source.

Dataset Storage Buckets are intentionally unaffected.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import runtime as base_runtime
from profiles import canonical_run_id, resolve_presets
from trainer.model_artifact import (
    download_verified_model_artifact,
    publish_verified_model_artifact,
)

_ORIGINAL_TRAINER_COMMAND = base_runtime._trainer_command
_ORIGINAL_HF_BUCKET_STORE = base_runtime._hf_bucket_store
_ORIGINAL_WRITE_HF_TRANSPORT_MANIFEST = base_runtime._write_hf_transport_manifest
_ORIGINAL_RUNTIME_CONTRACT = base_runtime._runtime_contract
_ORIGINAL_ASSERT_CONTRACT = base_runtime._assert_contract
_ORIGINAL_RUN_TRAINING = base_runtime.run_training
_INSTALLED = False


def _model_repo_store():
    return base_runtime._hf_model_repo_store()


def _model_repo_id() -> str:
    return base_runtime._hf_model_repo_id()


def _assert_frozen_source(metadata: Mapping[str, object], *, source: str) -> None:
    """Reject a live remote checkpoint produced by a different trainer commit."""

    expected = os.environ.get("SMALL_LLM_MODAL_SOURCE_COMMIT")
    observed = metadata.get("source_commit")
    if (
        expected
        and isinstance(observed, str)
        and observed
        and observed != expected
    ):
        raise RuntimeError(
            f"{source} checkpoint was created by source commit {observed}; "
            f"checkout that frozen commit instead of resuming with {expected}"
        )


def _trainer_command_model_repo(*args: Any, **kwargs: Any) -> list[str]:
    """Reuse the frozen trainer command while swapping bucket CLI flags for repo flags."""

    command = _ORIGINAL_TRAINER_COMMAND(*args, **kwargs)
    rewritten: list[str] = []
    for item in command:
        if item == "--remote-checkpoint-bucket":
            rewritten.append("--remote-checkpoint-repo")
        elif item == "--remote-create-bucket":
            rewritten.append("--remote-create-repo")
        elif item == "hf-bucket-cross-provider-resume":
            rewritten.append("hf-model-repo-cross-provider-resume")
        else:
            rewritten.append(item)
    return rewritten


def _restore_hf_checkpoint_repo_first(run_id: str, run_dir: Path) -> dict[str, Any] | None:
    """Restore local -> model repo live pointer -> stable model -> legacy bucket."""

    checkpoint_dir = run_dir / "checkpoints"
    local_id, _ = base_runtime._latest_checkpoint(checkpoint_dir)
    if local_id is not None:
        return None

    repo_store = _model_repo_store()
    pointer = repo_store.read_json(f"run/{run_id}/latest.json")
    if pointer is not None:
        if not isinstance(pointer, Mapping):
            raise RuntimeError("Hugging Face model-repository latest pointer is not a JSON object")
        metadata = base_runtime._restore_two_phase_pointer(
            store=repo_store,
            run_id=run_id,
            run_dir=run_dir,
            pointer=pointer,
            source="hf_model_repo",
            expected_transport="modal-hf-checkpoint-v1",
        )
        _assert_frozen_source(metadata, source="Hugging Face model-repository")
        return metadata

    # Stable artifacts are also valid exact checkpoint snapshots. This supports
    # the completed 100M/2B model that was manually moved from the bucket to
    # models/<run_id>/<step> before this transport was standardized.
    try:
        restored, info = download_verified_model_artifact(
            repo_id=_model_repo_id(),
            run_id=run_id,
            token=base_runtime._hf_token(),
            revision=None,
            destination=checkpoint_dir,
        )
    except RuntimeError as error:
        if "contains no artifact" not in str(error) and "pointer is missing" not in str(error):
            raise
    else:
        transport: Mapping[str, object] = {}
        transport_path = restored / "drive_manifest.json"
        if transport_path.is_file():
            transport = base_runtime._json(transport_path)
        metadata = base_runtime._verified_checkpoint_metadata(restored, restored.name)
        metadata.update(
            source="hf_model_artifact",
            source_commit=info.get("source_commit") or transport.get("source_commit"),
            microbatch_size=info.get("microbatch_size") or transport.get("microbatch_size"),
        )
        _assert_frozen_source(metadata, source="Hugging Face stable model")
        print(json.dumps({"hf_checkpoint_restore": metadata}, sort_keys=True), flush=True)
        return metadata

    # Legacy migration source only. New writes never target this bucket.
    bucket_store = _ORIGINAL_HF_BUCKET_STORE()
    legacy_pointer = bucket_store.read_json(f"run/{run_id}/latest.json")
    if legacy_pointer is None:
        return None
    if not isinstance(legacy_pointer, Mapping):
        raise RuntimeError("legacy Hugging Face bucket latest pointer is not a JSON object")
    return base_runtime._restore_two_phase_pointer(
        store=bucket_store,
        run_id=run_id,
        run_dir=run_dir,
        pointer=legacy_pointer,
        source="hf_bucket",
        expected_transport="modal-hf-bucket-checkpoint-v1",
    )


def _write_hf_transport_manifest_model_repo(
    path: Path,
    *,
    run_id: str,
    dataset: Path,
    dataset_profile: str,
    source_commit: str,
    microbatch_size: int,
    resume_parent_source_commit: str | None,
    bucket_id: str,
) -> dict[str, Any]:
    """Preserve the runtime call shape while recording the new repository transport."""

    payload = _ORIGINAL_WRITE_HF_TRANSPORT_MANIFEST(
        path,
        run_id=run_id,
        dataset=dataset,
        dataset_profile=dataset_profile,
        source_commit=source_commit,
        microbatch_size=microbatch_size,
        resume_parent_source_commit=resume_parent_source_commit,
        bucket_id=bucket_id,
    )
    payload = dict(payload)
    payload["transport"] = "modal-hf-checkpoint-v1"
    payload["repo_id"] = bucket_id
    payload.pop("bucket_id", None)
    payload["retention"] = "latest_only_model_repo_history_squashed"
    if payload.get("source_migration"):
        payload["source_migration"] = (
            "legacy HF checkpoint transport -> unified HF model-repository runtime"
        )
    base_runtime._write_json(path, payload)
    return payload


def _runtime_contract_model_repo(*args: Any, **kwargs: Any) -> dict[str, Any]:
    result = dict(_ORIGINAL_RUNTIME_CONTRACT(*args, **kwargs))
    repo_id = kwargs.get("bucket_id")
    result["checkpoint_transports"] = {
        "local": f"Modal Volume every {base_runtime.DURABILITY_EVERY} successful updates",
        "remote": (
            f"Hugging Face model repository {repo_id} every "
            f"{base_runtime.HF_REMOTE_EVERY} successful updates plus final"
        ),
    }
    if result.get("source_migration"):
        result["source_migration"] = "checkpoint transport unified on HF model repository"
    return result


def _assert_contract_model_repo(path: Path, expected: Mapping[str, Any]) -> None:
    """Keep scientific immutables frozen while allowing this infrastructure migration."""

    actual = base_runtime._json(path)
    actual_source = actual.get("source_commit")
    expected_source = expected.get("source_commit")
    migration_parent = os.environ.get(
        "SMALL_LLM_INFRA_MIGRATION_PARENT_COMMIT", ""
    ).strip()
    if actual_source != expected_source and actual_source == migration_parent:
        compatible = dict(expected)
        compatible["source_commit"] = actual_source
        _ORIGINAL_ASSERT_CONTRACT(path, compatible)
        actual["source_commit"] = expected_source
        actual["resume_parent_source_commit"] = actual_source
        actual["source_migration"] = (
            "Beam step-250 checkpoint fsync infrastructure-only migration"
        )
        base_runtime._write_json(path, actual)
    else:
        _ORIGINAL_ASSERT_CONTRACT(path, expected)
    actual = base_runtime._json(path)
    desired = expected.get("checkpoint_transports")
    if actual.get("checkpoint_transports") != desired:
        actual["checkpoint_transports"] = desired
        actual["checkpoint_transport_migration"] = (
            "ADR-0055: HF Storage Bucket -> unified HF model repository"
        )
        base_runtime._write_json(path, actual)


def _publish_final_model_if_complete(
    *,
    model: str,
    tokens: str,
    run_root: Path,
    result: Mapping[str, object],
) -> dict[str, object] | None:
    completed = result.get("completed_steps")
    total = result.get("total_steps")
    if (
        isinstance(completed, bool)
        or isinstance(total, bool)
        or not isinstance(completed, int)
        or not isinstance(total, int)
        or completed != total
    ):
        return None

    model_preset, token_preset = resolve_presets(model, tokens)
    run_id = canonical_run_id(model_preset, token_preset)
    run_dir = run_root / run_id
    checkpoint_id, step = base_runtime._latest_checkpoint(run_dir / "checkpoints")
    if checkpoint_id is None or step != total:
        raise RuntimeError("completed Modal run has no matching verified final checkpoint")
    checkpoint_root = run_dir / "checkpoints" / checkpoint_id
    runtime = base_runtime._json(run_dir / "modal_runtime.json")

    metadata = {
        "artifact_type": "small-llm-final-joint-checkpoint",
        "model_label": model_preset.label,
        "model_parameters_nominal": model_preset.parameters,
        "trainer_model_size": model_preset.trainer_size,
        "token_label": token_preset.label,
        "training_tokens_nominal": token_preset.tokens,
        "dataset_profile": token_preset.dataset_profile,
        "completed_steps": step,
        "total_steps": total,
        "is_final": True,
        "source_commit": runtime.get("source_commit"),
        "dataset_run_id": runtime.get("dataset_run_id"),
        "precision": runtime.get("precision"),
        "microbatch_size": runtime.get("microbatch_size"),
    }
    return publish_verified_model_artifact(
        repo_id=_model_repo_id(),
        run_id=run_id,
        checkpoint_root=checkpoint_root,
        token=base_runtime._hf_token(),
        metadata=metadata,
    )


def run_training(**kwargs: Any) -> dict[str, object]:
    """Run the frozen Modal trainer with unified model-repository checkpoint transport."""

    install_model_repo_checkpoint_transport()
    os.environ["SMALL_LLM_MODAL_SOURCE_COMMIT"] = str(kwargs["source_commit"])
    result = dict(_ORIGINAL_RUN_TRAINING(**kwargs))
    result["hf_checkpoint_transport"] = {
        "repo": _model_repo_id(),
        "cadence_steps": base_runtime.HF_REMOTE_EVERY,
        "retention": "latest_only_model_repo_history_squashed",
    }
    artifact = _publish_final_model_if_complete(
        model=str(kwargs["model"]),
        tokens=str(kwargs["tokens"]),
        run_root=Path(kwargs["run_root"]),
        result=result,
    )
    if artifact is not None:
        result["hf_model_artifact"] = artifact
    return result


def install_model_repo_checkpoint_transport() -> None:
    """Install the transport-only overrides into the imported Modal runtime."""

    global _INSTALLED
    if _INSTALLED:
        return
    base_runtime._trainer_command = _trainer_command_model_repo
    base_runtime._restore_hf_checkpoint_if_needed = _restore_hf_checkpoint_repo_first
    # The legacy runtime calls this accessor and labels its return value bucket_id;
    # after this adapter is installed the value is intentionally the model repo ID.
    base_runtime._hf_checkpoint_bucket_id = _model_repo_id
    # rolling_dataset.next_unconsumed_block calls this historical accessor. Make
    # it read the model-repository run/latest pointer without changing dataset buckets.
    base_runtime._hf_bucket_store = _model_repo_store
    base_runtime._write_hf_transport_manifest = _write_hf_transport_manifest_model_repo
    base_runtime._runtime_contract = _runtime_contract_model_repo
    base_runtime._assert_contract = _assert_contract_model_repo
    base_runtime.run_training = run_training
    _INSTALLED = True


__all__ = ["install_model_repo_checkpoint_transport", "run_training"]
