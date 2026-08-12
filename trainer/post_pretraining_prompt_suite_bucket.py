"""Run the standard post-pretraining suite from a Modal HF Storage Bucket checkpoint.

Modal production uses rolling latest-only checkpoint retention in a private
Hugging Face Storage Bucket. This entrypoint reuses the normal prompt suite
unchanged while swapping only its checkpoint transport.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Sequence

from dataset.src.hf_bucket_checkpoint import HuggingFaceBucketCheckpointStore
from dataset.src.joint_checkpoint import (
    _verify_published_checkpoint_manifest,
    verify_local_manifest,
)
from trainer import post_pretraining_prompt_suite as suite


def _resolve_bucket_id(repo_id: str) -> str:
    """Return the configured Modal checkpoint bucket or its canonical default."""

    override = os.environ.get("SMALL_LLM_HF_CHECKPOINT_BUCKET_ID")
    if override is not None and override.strip():
        return override.strip()
    return f"{repo_id}-checkpoints"


def download_verified_bucket_checkpoint(
    *,
    repo_id: str,
    run_id: str | None,
    token: str | None,
    revision: str | None,
    pointer_name: str,
    destination: Path,
) -> tuple[Path, dict[str, object]]:
    """Download and verify the rolling latest checkpoint from the Modal bucket."""

    if pointer_name != "latest":
        raise RuntimeError(
            "Modal Hugging Face Storage Bucket checkpoints use rolling latest-only "
            "retention; run this entrypoint with --pointer latest"
        )
    if run_id is None or not run_id.strip():
        raise RuntimeError(
            "Modal bucket evaluation requires --run-id or SMALL_LLM_RUN_ID"
        )
    if revision is not None:
        raise RuntimeError(
            "--revision applies to Git-backed Hugging Face model repositories and "
            "is not supported for Storage Bucket checkpoints"
        )

    selected_run_id = run_id.strip()
    bucket_id = _resolve_bucket_id(repo_id)
    store = HuggingFaceBucketCheckpointStore(
        bucket_id,
        token=token,
        private=True,
    )
    pointer_path = f"run/{selected_run_id}/latest.json"
    pointer = store.read_json(pointer_path)
    if pointer is None:
        raise RuntimeError(
            f"Hugging Face Storage Bucket pointer is missing: "
            f"{bucket_id}/{pointer_path}"
        )

    checkpoint_id, prefix = suite._checkpoint_prefix(
        pointer,
        run_id=selected_run_id,
        pointer_name="latest",
    )
    checkpoint_root = destination / checkpoint_id
    checkpoint_root.mkdir(parents=True, exist_ok=False)
    store.download_tree(prefix, checkpoint_root)
    verify_local_manifest(checkpoint_root)

    embedded_manifest = suite._json_object(
        checkpoint_root / "checkpoint_manifest.json",
        label="checkpoint_manifest.json",
    )
    pointer_manifest = pointer.get("checkpoint_manifest")
    supplied_manifest = (
        pointer_manifest if isinstance(pointer_manifest, Mapping) else embedded_manifest
    )
    _verify_published_checkpoint_manifest(checkpoint_root, supplied_manifest)

    return checkpoint_root, {
        "checkpoint_source": "hf_storage_bucket",
        "bucket_id": bucket_id,
        "repo_id": repo_id,
        "run_id": selected_run_id,
        "pointer": "latest",
        "checkpoint_id": checkpoint_id,
        "prefix": prefix,
        "metric": pointer.get("metric"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the canonical suite CLI with only checkpoint transport replaced."""

    original = suite.download_verified_checkpoint
    suite.download_verified_checkpoint = download_verified_bucket_checkpoint
    try:
        return suite.main(argv)
    finally:
        suite.download_verified_checkpoint = original


if __name__ == "__main__":
    raise SystemExit(main())
