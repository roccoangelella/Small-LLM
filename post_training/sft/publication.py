"""Compatibility identity envelope for publishing SFT checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from dataset.src.remote import sha256_path

from .bundle import verify_bundle


def publication_dataset_manifest(
    bundle_root: Path | str,
    *,
    run_id: str,
) -> dict[str, object]:
    """Describe a verified immutable SFT bundle for the generic checkpoint publisher.

    The generic publisher historically names this object ``drive_manifest``.
    For SFT the durable source is the attached immutable SFT bundle rather than
    the pretraining Google-Drive shard transport. The provider field makes that
    distinction explicit while preserving the already-qualified publisher.
    """

    root = Path(bundle_root)
    verification = verify_bundle(root)
    bundle_hash = verification["bundle_manifest_sha256"]
    shards: list[dict[str, object]] = []
    for split in ("train", "validation", "test"):
        split_root = root / split
        manifest = json.loads((split_root / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, Mapping):
            raise RuntimeError(f"invalid SFT {split} manifest")
        raw_shards = manifest.get("shards")
        if not isinstance(raw_shards, list):
            raise RuntimeError(f"SFT {split} manifest has no shards")
        for item in raw_shards:
            if not isinstance(item, Mapping):
                raise RuntimeError(f"SFT {split} shard entry is malformed")
            name = item.get("path")
            digest = item.get("sha256")
            byte_size = item.get("byte_size")
            if (
                not isinstance(name, str)
                or not isinstance(digest, str)
                or not isinstance(byte_size, int)
            ):
                raise RuntimeError(f"SFT {split} shard identity is malformed")
            path = split_root / name
            if sha256_path(path) != digest or path.stat().st_size != byte_size:
                raise RuntimeError(f"SFT {split} shard verification drifted: {name}")
            shards.append(
                {
                    "filename": f"{split}/{name}",
                    "byte_size": byte_size,
                    "local_sha256": digest,
                    "checksum": digest,
                    "remote_durable": True,
                    "provider": "verified_immutable_sft_bundle",
                    "bundle_manifest_sha256": bundle_hash,
                }
            )
    return {
        "version": 1,
        "run_id": run_id,
        "configuration_hash": bundle_hash,
        "schema_hash": bundle_hash,
        "transport": "verified_immutable_sft_bundle",
        "shards": shards,
    }


__all__ = ["publication_dataset_manifest"]
