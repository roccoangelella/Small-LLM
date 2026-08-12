"""Hugging Face Storage Bucket backend for immutable dataset shards.

The dataset bucket is an object store, not a trainer cache.  Finalized schema-v2
shards are uploaded under ``run/<dataset_run_id>/<logical_name>`` and are never
overwritten with different bytes.  Every upload is read back and SHA-256 checked
before it is considered durable.  The completed dataset manifest is published
last together with a small readiness pointer.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .remote import (
    _prepare_download_paths,
    _safe_relative_path,
    ensure_safe_directory,
    safe_path_component,
    sha256_path,
)


class HuggingFaceBucketShardStore:
    """RemoteShardStore-compatible immutable shard store backed by an HF bucket."""

    def __init__(
        self,
        bucket_id: str,
        *,
        token: str | None = None,
        private: bool = True,
        api: Any | None = None,
        create_bucket: bool = False,
    ) -> None:
        if not isinstance(bucket_id, str) or not bucket_id.strip():
            raise RuntimeError("Hugging Face dataset bucket ID cannot be empty")
        self.bucket_id = bucket_id.strip()
        self.token = token
        self.private = bool(private)
        if api is None:
            try:
                from huggingface_hub import HfApi
            except ImportError as error:
                raise RuntimeError(
                    "Hugging Face dataset shards require huggingface_hub>=1.5"
                ) from error
            api = HfApi(token=token)
        self.api = api
        if create_bucket:
            self.ensure_bucket()

    def _kwargs(self) -> dict[str, object]:
        return {} if self.token is None else {"token": self.token}

    def _require_method(self, name: str):
        method = getattr(self.api, name, None)
        if not callable(method):
            raise RuntimeError(
                f"configured huggingface_hub lacks HfApi.{name}; require huggingface_hub>=1.5"
            )
        return method

    def ensure_bucket(self) -> None:
        self._require_method("create_bucket")(
            bucket_id=self.bucket_id,
            private=self.private,
            exist_ok=True,
            **self._kwargs(),
        )

    @staticmethod
    def object_key(run_id: str, logical_name: str) -> str:
        run_id = safe_path_component(run_id, label="run_id")
        logical_name = _safe_relative_path(logical_name).as_posix()
        return f"run/{run_id}/{logical_name}"

    def _list_files(self, *, prefix: str | None = None) -> list[object]:
        items = self._require_method("list_bucket_tree")(
            bucket_id=self.bucket_id,
            prefix=prefix,
            recursive=True,
            **self._kwargs(),
        )
        return [
            item
            for item in items
            if getattr(item, "type", None) == "file"
            and isinstance(getattr(item, "path", None), str)
        ]

    def _exact_file(self, key: str) -> object | None:
        matches = [
            item for item in self._list_files(prefix=key)
            if getattr(item, "path", None) == key
        ]
        if len(matches) > 1:
            raise RuntimeError(f"Hugging Face bucket returned duplicate object metadata: {key}")
        return matches[0] if matches else None

    def _download_object(self, item: object, destination: Path) -> None:
        ensure_safe_directory(destination.parent)
        self._require_method("download_bucket_files")(
            bucket_id=self.bucket_id,
            files=[(item, str(destination))],
            raise_on_missing_files=True,
            **self._kwargs(),
        )

    def upload_finalized_shard(
        self,
        *,
        run_id: str,
        logical_name: str,
        local_path: Path,
        resume_state: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        del resume_state
        if local_path.suffix != ".bin" or ".tmp" in local_path.name:
            raise ValueError("only finalized immutable .bin shards may be uploaded")
        key = self.object_key(run_id, logical_name)
        size = local_path.stat().st_size
        digest = sha256_path(local_path)
        existing = self._exact_file(key)
        if existing is not None:
            remote_size = getattr(existing, "size", None)
            if remote_size is not None and int(remote_size) != size:
                raise RuntimeError("refusing to overwrite immutable HF dataset shard with a different size")
            verified = self.verify_remote_shard(
                run_id=run_id,
                logical_name=logical_name,
                file_id=key,
                byte_size=size,
                sha256=digest,
            )
            return {
                "file_id": key,
                "size": size,
                "sha256": digest,
                "upload_complete": True,
                "verified_existing": True,
                **verified,
            }

        before = digest
        self._require_method("batch_bucket_files")(
            bucket_id=self.bucket_id,
            add=[(str(local_path), key)],
            **self._kwargs(),
        )
        if sha256_path(local_path) != before:
            raise RuntimeError(f"local shard changed during Hugging Face bucket upload: {local_path}")
        return {
            "file_id": key,
            "size": size,
            "sha256": digest,
            "upload_complete": True,
        }

    def verify_remote_shard(
        self,
        *,
        run_id: str,
        logical_name: str,
        file_id: str,
        byte_size: int,
        sha256: str,
    ) -> dict[str, object]:
        key = self.object_key(run_id, logical_name)
        if file_id != key:
            raise RuntimeError("HF dataset shard object key does not match its logical name")
        item = self._exact_file(key)
        if item is None:
            raise RuntimeError(f"Hugging Face dataset shard is missing: {key}")
        remote_size = getattr(item, "size", None)
        if remote_size is not None and int(remote_size) != byte_size:
            raise RuntimeError(f"Hugging Face dataset shard size mismatch: {key}")
        with tempfile.TemporaryDirectory(prefix="small-llm-hf-dataset-verify-") as tmp:
            target = Path(tmp) / "shard.bin"
            self._download_object(item, target)
            if target.stat().st_size != byte_size or sha256_path(target) != sha256:
                raise RuntimeError(f"Hugging Face dataset shard read-back mismatch: {key}")
        return {
            "file_id": key,
            "size": byte_size,
            "sha256": sha256,
            "sha256_source": "downloaded HF bucket read-back",
            "verified_at": "hf-bucket-readback",
        }

    def download_shard(
        self,
        *,
        run_id: str,
        logical_name: str,
        file_id: str,
        destination: Path,
        byte_size: int,
        sha256: str,
    ) -> None:
        key = self.object_key(run_id, logical_name)
        if file_id != key:
            raise RuntimeError("HF dataset shard object key does not match its logical name")
        item = self._exact_file(key)
        if item is None:
            raise RuntimeError(f"Hugging Face dataset shard is missing: {key}")
        destination, part = _prepare_download_paths(destination)
        ensure_safe_directory(destination.parent)
        part.unlink(missing_ok=True)
        self._download_object(item, part)
        if part.stat().st_size != byte_size or sha256_path(part) != sha256:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"downloaded HF dataset shard checksum or size mismatch: {key}")
        os.replace(part, destination)

    def list_manifest_entries(self, run_id: str) -> list[dict[str, object]]:
        run_id = safe_path_component(run_id, label="run_id")
        prefix = f"run/{run_id}/"
        result: list[dict[str, object]] = []
        for item in self._list_files(prefix=prefix):
            key = str(getattr(item, "path"))
            if not key.endswith(".bin"):
                continue
            result.append(
                {
                    "logical_name": key[len(prefix):],
                    "file_id": key,
                    "size": int(getattr(item, "size", 0)),
                }
            )
        return result

    resume_upload = upload_finalized_shard
    resume_download = download_shard

    def _write_json(self, key: str, payload: Mapping[str, object]) -> None:
        _safe_relative_path(key)
        data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        self._require_method("batch_bucket_files")(
            bucket_id=self.bucket_id,
            add=[(data, key)],
            **self._kwargs(),
        )

    def _read_json(self, key: str) -> dict[str, object] | None:
        item = self._exact_file(key)
        if item is None:
            return None
        with tempfile.TemporaryDirectory(prefix="small-llm-hf-dataset-json-") as tmp:
            target = Path(tmp) / "object.json"
            self._download_object(item, target)
            payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise RuntimeError(f"Hugging Face dataset JSON object is not a mapping: {key}")
        return dict(payload)

    def publish_dataset_manifest(self, *, run_id: str, manifest_path: Path) -> dict[str, object]:
        run_id = safe_path_component(run_id, label="run_id")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, Mapping):
            raise RuntimeError("dataset manifest must be a JSON object")
        production = manifest.get("production")
        if not isinstance(production, Mapping) or production.get("run_id") != run_id:
            raise RuntimeError("dataset manifest run ID does not match HF dataset bucket run")
        shards = manifest.get("shards")
        if not isinstance(shards, list) or not shards:
            raise RuntimeError("dataset manifest has no shard inventory")

        key = self.object_key(run_id, "manifest.json")
        digest = sha256_path(manifest_path)
        self._require_method("batch_bucket_files")(
            bucket_id=self.bucket_id,
            add=[(str(manifest_path), key)],
            **self._kwargs(),
        )
        item = self._exact_file(key)
        if item is None:
            raise RuntimeError("published HF dataset manifest is missing")
        with tempfile.TemporaryDirectory(prefix="small-llm-hf-dataset-manifest-") as tmp:
            downloaded = Path(tmp) / "manifest.json"
            self._download_object(item, downloaded)
            if sha256_path(downloaded) != digest:
                raise RuntimeError("published HF dataset manifest read-back mismatch")

        train = sum(1 for row in shards if isinstance(row, Mapping) and row.get("split") == "train")
        validation = sum(
            1 for row in shards if isinstance(row, Mapping) and row.get("split") == "validation"
        )
        ready = {
            "version": 1,
            "run_id": run_id,
            "manifest_sha256": digest,
            "train_shards": train,
            "validation_shards": validation,
            "target_reached": production.get("target_reached") is True,
        }
        ready_key = self.object_key(run_id, "ready.json")
        self._write_json(ready_key, ready)
        if self._read_json(ready_key) != ready:
            raise RuntimeError("HF dataset readiness pointer read-back mismatch")
        return ready

    def download_dataset_manifest(self, *, run_id: str, destination: Path) -> dict[str, object]:
        run_id = safe_path_component(run_id, label="run_id")
        ready_key = self.object_key(run_id, "ready.json")
        ready = self._read_json(ready_key)
        if ready is None or ready.get("run_id") != run_id or ready.get("target_reached") is not True:
            raise RuntimeError(f"HF dataset run is not marked ready: {run_id}")
        digest = ready.get("manifest_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError("HF dataset readiness pointer has an invalid manifest hash")
        key = self.object_key(run_id, "manifest.json")
        item = self._exact_file(key)
        if item is None:
            raise RuntimeError(f"HF dataset manifest is missing: {run_id}")
        ensure_safe_directory(destination.parent)
        self._download_object(item, destination)
        if sha256_path(destination) != digest:
            destination.unlink(missing_ok=True)
            raise RuntimeError("downloaded HF dataset manifest checksum mismatch")
        payload = json.loads(destination.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise RuntimeError("downloaded HF dataset manifest is not a JSON object")
        production = payload.get("production")
        if not isinstance(production, Mapping) or production.get("run_id") != run_id:
            raise RuntimeError("downloaded HF dataset manifest run ID mismatch")
        return dict(payload)


__all__ = ["HuggingFaceBucketShardStore"]
