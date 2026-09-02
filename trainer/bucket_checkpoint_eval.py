"""Evaluation download helper for live checkpoints stored in HF Storage Buckets."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from dataset.src.hf_bucket_checkpoint import HuggingFaceBucketCheckpointStore
from dataset.src.joint_checkpoint import (
    _verify_published_checkpoint_manifest,
    verify_local_manifest,
)
from trainer.post_pretraining_prompt_suite import _checkpoint_prefix, _json_object


def _discover_run_id(
    store: HuggingFaceBucketCheckpointStore,
    *,
    pointer: str,
) -> str:
    """Return the unique run id with a matching bucket pointer."""

    matches: set[str] = set()
    for item in store._list_files(prefix="run/"):
        path = getattr(item, "path", None)
        if not isinstance(path, str):
            continue
        parts = path.split("/")
        if len(parts) == 3 and parts[0] == "run" and parts[2] == f"{pointer}.json":
            matches.add(parts[1])
    if not matches:
        raise RuntimeError(f"the bucket contains no run/*/{pointer}.json pointer")
    if len(matches) != 1:
        raise RuntimeError(
            f"the bucket contains multiple {pointer} pointers; pass --run-id explicitly: {sorted(matches)}"
        )
    return next(iter(matches))


def download_verified_checkpoint(
    *,
    repo_id: str,
    run_id: str | None,
    token: str | None,
    revision: str | None,
    pointer_name: str,
    destination: Path,
) -> tuple[Path, dict[str, object]]:
    """Download and verify one best/latest checkpoint tree from a HF Storage Bucket.

    The existing evaluation CLIs pass the storage bucket id through ``--repo-id``
    so older command shapes remain usable. HF Storage Buckets are non-versioned,
    therefore ``--revision`` is intentionally rejected here.
    """

    if revision is not None:
        raise RuntimeError("HF Storage Bucket checkpoints are non-versioned; omit --revision")

    store = HuggingFaceBucketCheckpointStore(
        repo_id,
        token=token,
        private=True,
    )
    selected_run_id = run_id or _discover_run_id(store, pointer=pointer_name)
    pointer_path = f"run/{selected_run_id}/{pointer_name}.json"
    pointer = store.read_json(pointer_path)
    if pointer is None:
        raise RuntimeError(f"Hugging Face bucket pointer is missing: {pointer_path}")
    checkpoint_id, prefix = _checkpoint_prefix(
        pointer,
        run_id=selected_run_id,
        pointer_name=pointer_name,
    )
    checkpoint_root = destination / checkpoint_id
    checkpoint_root.mkdir(parents=True, exist_ok=False)
    store.download_tree(prefix, checkpoint_root)
    verify_local_manifest(checkpoint_root)
    embedded_manifest = _json_object(
        checkpoint_root / "checkpoint_manifest.json",
        label="checkpoint_manifest.json",
    )
    pointer_manifest = pointer.get("checkpoint_manifest")
    supplied_manifest = (
        pointer_manifest if isinstance(pointer_manifest, Mapping) else embedded_manifest
    )
    _verify_published_checkpoint_manifest(checkpoint_root, supplied_manifest)
    return checkpoint_root, {
        "transport": "hf_storage_bucket",
        "repo_id": repo_id,
        "bucket_id": repo_id,
        "run_id": selected_run_id,
        "pointer": pointer_name,
        "checkpoint_id": checkpoint_id,
        "prefix": prefix,
        "metric": pointer.get("metric"),
    }


__all__ = ["download_verified_checkpoint"]
