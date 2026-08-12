"""Canonical Hugging Face model-artifact helpers for completed Small-LLM checkpoints.

Live trainer checkpoints may use the two-phase ``run/<run_id>/...`` namespace.
Stable human-facing artifacts use ``models/<run_id>/<checkpoint_id>`` plus
``models/<run_id>/artifact.json``. These helpers make the latter transport
provider-neutral for evaluation and Modal final publication.

Stable artifacts are native saved checkpoints. Their integrity contract is
``local_manifest.json``; ``checkpoint_manifest.json`` is publication metadata
specific to the two-phase live ``run/...`` protocol and is not required here.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Mapping

from dataset.src.joint_checkpoint import verify_local_manifest
from dataset.src.remote import HuggingFaceCheckpointStore

_CHECKPOINT_ID = re.compile(r"^step-(\d{8})$")


def _valid_checkpoint_id(value: object) -> str:
    if not isinstance(value, str) or _CHECKPOINT_ID.fullmatch(value) is None:
        raise RuntimeError(f"invalid model artifact checkpoint_id: {value!r}")
    return value


def resolve_model_artifact(
    store: HuggingFaceCheckpointStore,
    *,
    run_id: str,
) -> tuple[str, str, dict[str, object]]:
    """Resolve the stable artifact pointer, or discover the newest model directory.

    Discovery is a compatibility path for a manually moved checkpoint that has
    already been placed under ``models/<run_id>/step-XXXXXXXX`` but does not yet
    have ``artifact.json``.
    """

    if not isinstance(run_id, str) or not run_id.strip() or "/" in run_id or "\\" in run_id:
        raise RuntimeError(f"invalid model artifact run_id: {run_id!r}")
    run_id = run_id.strip()
    metadata_path = f"models/{run_id}/artifact.json"
    metadata = store.read_json(metadata_path)
    if metadata is not None:
        if not isinstance(metadata, Mapping):
            raise RuntimeError(f"{metadata_path} is not a JSON object")
        info = dict(metadata)
        checkpoint_id = _valid_checkpoint_id(info.get("checkpoint_id"))
        expected = f"models/{run_id}/{checkpoint_id}"
        prefix = info.get("huggingface_path", expected)
        if prefix != expected:
            raise RuntimeError(
                f"{metadata_path} path disagrees with its run/checkpoint identity"
            )
        return checkpoint_id, expected, info

    list_files = getattr(store.api, "list_repo_files", None)
    if not callable(list_files):
        raise RuntimeError(
            f"Hugging Face model artifact pointer is missing: {metadata_path}"
        )
    root = f"models/{run_id}/"
    candidates: dict[str, int] = {}
    for name in list_files(**store._hub_kwargs()):
        if not isinstance(name, str) or not name.startswith(root):
            continue
        remainder = name[len(root) :]
        checkpoint_id = remainder.split("/", 1)[0]
        match = _CHECKPOINT_ID.fullmatch(checkpoint_id)
        if match is not None and "/" in remainder:
            candidates[checkpoint_id] = int(match.group(1))
    if not candidates:
        raise RuntimeError(
            f"Hugging Face model repository contains no artifact for run {run_id!r}"
        )
    checkpoint_id = max(candidates, key=candidates.get)
    prefix = f"models/{run_id}/{checkpoint_id}"
    return checkpoint_id, prefix, {
        "version": 1,
        "artifact_type": "small-llm-discovered-joint-checkpoint",
        "run_id": run_id,
        "checkpoint_id": checkpoint_id,
        "huggingface_path": prefix,
        "pointer_discovered": True,
    }


def download_verified_model_artifact(
    *,
    repo_id: str,
    run_id: str,
    token: str | None,
    revision: str | None,
    destination: Path,
) -> tuple[Path, dict[str, object]]:
    """Download and verify one stable native model-repository checkpoint.

    Stable ``models/...`` artifacts are verified with the checkpoint's native
    ``local_manifest.json``. The publisher-level ``checkpoint_manifest.json``
    used by two-phase live ``run/...`` checkpoints is deliberately not required.
    """

    store = HuggingFaceCheckpointStore(
        repo_id,
        token=token,
        private=True,
        revision=revision,
    )
    checkpoint_id, prefix, metadata = resolve_model_artifact(store, run_id=run_id)
    checkpoint_root = destination / checkpoint_id
    if checkpoint_root.exists() or checkpoint_root.is_symlink():
        raise FileExistsError(f"model artifact destination already exists: {checkpoint_root}")
    checkpoint_root.mkdir(parents=True, exist_ok=False)
    try:
        store.download_tree(prefix, checkpoint_root)
        verify_local_manifest(checkpoint_root)
    except BaseException:
        shutil.rmtree(checkpoint_root, ignore_errors=True)
        raise
    return checkpoint_root, {
        "checkpoint_source": "hf_model_artifact",
        "repo_id": repo_id,
        "run_id": run_id,
        "checkpoint_id": checkpoint_id,
        "prefix": prefix,
        **metadata,
    }


def publish_verified_model_artifact(
    *,
    repo_id: str,
    run_id: str,
    checkpoint_root: Path,
    token: str | None,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    """Publish a verified checkpoint under the canonical stable-model namespace."""

    from huggingface_hub import HfApi

    verify_local_manifest(checkpoint_root)
    checkpoint_id = _valid_checkpoint_id(checkpoint_root.name)
    prefix = f"models/{run_id}/{checkpoint_id}"
    metadata_path = f"models/{run_id}/artifact.json"
    payload = {
        "version": 1,
        **dict(metadata),
        "run_id": run_id,
        "checkpoint_id": checkpoint_id,
        "huggingface_path": prefix,
        "verification": "dataset.src.joint_checkpoint.verify_local_manifest passed before upload",
    }

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    folder_commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=checkpoint_root,
        path_in_repo=prefix,
        commit_message=f"Publish {run_id} model artifact {checkpoint_id}",
    )
    metadata_commit = api.upload_file(
        repo_id=repo_id,
        repo_type="model",
        path_or_fileobj=(json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        path_in_repo=metadata_path,
        commit_message=f"Point {run_id} model artifact to {checkpoint_id}",
    )
    return {
        "status": "published",
        "repo_id": repo_id,
        "run_id": run_id,
        "checkpoint_id": checkpoint_id,
        "path_in_repo": prefix,
        "metadata_path": metadata_path,
        "checkpoint_commit": getattr(folder_commit, "oid", None),
        "metadata_commit": getattr(metadata_commit, "oid", None),
    }


__all__ = [
    "download_verified_model_artifact",
    "publish_verified_model_artifact",
    "resolve_model_artifact",
]
