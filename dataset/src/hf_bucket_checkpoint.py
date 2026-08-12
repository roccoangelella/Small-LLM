"""Hugging Face Storage Bucket backend for verified training checkpoints.

Buckets are deliberately non-versioned mutable object storage.  The existing
TwoPhaseCheckpointPublisher still owns checkpoint-manifest construction and the
latest-pointer move; this store supplies byte-verified upload/download primitives
without creating Git history for each checkpoint boundary.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .remote import (
    _safe_download_target,
    _safe_relative_path,
    _tree_files,
    ensure_safe_directory,
    safe_path_component,
    sha256_path,
)


class HuggingFaceBucketCheckpointStore:
    """RemoteCheckpointStore backed by a private Hugging Face Storage Bucket."""

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
            raise RuntimeError("Hugging Face checkpoint bucket ID cannot be empty")
        self.bucket_id = bucket_id.strip()
        self.token = token
        self.private = bool(private)
        if api is None:
            try:
                from huggingface_hub import HfApi
            except ImportError as error:
                raise RuntimeError(
                    "Hugging Face Storage Bucket checkpoints require huggingface_hub>=1.5"
                ) from error
            api = HfApi(token=token)
        self.api = api
        if create_bucket:
            self.ensure_bucket()

    def _kwargs(self) -> dict[str, object]:
        return {} if self.token is None else {"token": self.token}

    @staticmethod
    def _remote_prefix(remote_prefix: str) -> str:
        prefix = remote_prefix.rstrip("/")
        if not prefix:
            raise RuntimeError("Hugging Face checkpoint bucket prefix cannot be empty")
        return _safe_relative_path(prefix).as_posix()

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
            if getattr(item, "type", None) == "file" and isinstance(getattr(item, "path", None), str)
        ]

    def _download_files(self, files: list[tuple[object, Path]]) -> None:
        if not files:
            return
        self._require_method("download_bucket_files")(
            bucket_id=self.bucket_id,
            files=[(source, str(target)) for source, target in files],
            raise_on_missing_files=True,
            **self._kwargs(),
        )

    def upload_tree(self, remote_prefix: str, local_dir: Path) -> dict[str, str]:
        prefix = self._remote_prefix(remote_prefix)
        local_files = _tree_files(local_dir)
        before: dict[Path, str] = {path: sha256_path(path) for path in local_files}
        additions: list[tuple[str, str]] = []
        remote_keys: list[str] = []
        for path in local_files:
            relative = path.relative_to(local_dir).as_posix()
            key = f"{prefix}/{relative}"
            _safe_relative_path(key)
            additions.append((str(path), key))
            remote_keys.append(key)

        if additions:
            self._require_method("batch_bucket_files")(
                bucket_id=self.bucket_id,
                add=additions,
                **self._kwargs(),
            )

        for path, digest in before.items():
            if sha256_path(path) != digest:
                raise RuntimeError(f"local file changed during Hugging Face bucket upload: {path}")

        # The bucket API is non-transactional and does not independently attest
        # the uploaded byte stream. Read every checkpoint object back before the
        # two-phase publisher is allowed to move latest.json.
        infos = {
            getattr(item, "path"): item
            for item in self._list_files(prefix=prefix + "/")
            if getattr(item, "path", None) in set(remote_keys)
        }
        missing = sorted(set(remote_keys) - set(infos))
        if missing:
            raise RuntimeError(f"Hugging Face bucket upload is missing checkpoint files: {missing}")

        result: dict[str, str] = {}
        with tempfile.TemporaryDirectory(prefix="small-llm-hf-bucket-verify-") as tmp:
            root = Path(tmp)
            downloads: list[tuple[object, Path]] = []
            for key in remote_keys:
                relative = _safe_relative_path(key[len(prefix) + 1 :])
                target = _safe_download_target(root, relative)
                ensure_safe_directory(target.parent)
                downloads.append((infos[key], target))
            self._download_files(downloads)
            for key, (_, target) in zip(remote_keys, downloads, strict=True):
                if target.is_symlink() or not target.is_file():
                    raise RuntimeError(f"Hugging Face bucket read-back is not a regular file: {key}")
                result[key] = sha256_path(target)
        return result

    def read_json(self, path: str) -> Mapping[str, object] | None:
        path = _safe_relative_path(path).as_posix()
        exact = [item for item in self._list_files(prefix=path) if getattr(item, "path", None) == path]
        if not exact:
            return None
        if len(exact) != 1:
            raise RuntimeError(f"Hugging Face bucket returned duplicate path metadata: {path}")
        with tempfile.TemporaryDirectory(prefix="small-llm-hf-bucket-json-") as tmp:
            target = Path(tmp) / "object.json"
            self._download_files([(exact[0], target)])
            payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise RuntimeError(f"Hugging Face bucket JSON object is not a mapping: {path}")
        return dict(payload)

    def write_json(self, path: str, value: Mapping[str, object]) -> None:
        path = _safe_relative_path(path).as_posix()
        payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        self._require_method("batch_bucket_files")(
            bucket_id=self.bucket_id,
            add=[(payload, path)],
            **self._kwargs(),
        )
        observed = self.read_json(path)
        if observed != dict(value):
            raise RuntimeError(f"Hugging Face bucket JSON read-back mismatch: {path}")

    def download_tree(self, remote_prefix: str, destination: Path) -> None:
        prefix = self._remote_prefix(remote_prefix)
        marker = prefix + "/"
        files = [
            item
            for item in self._list_files(prefix=marker)
            if str(getattr(item, "path")).startswith(marker)
        ]
        if not files:
            raise RuntimeError(f"Hugging Face bucket checkpoint prefix is missing: {prefix}")
        ensure_safe_directory(destination)
        downloads: list[tuple[object, Path]] = []
        for item in files:
            name = str(getattr(item, "path"))
            relative = _safe_relative_path(name[len(marker) :])
            target = _safe_download_target(destination, relative)
            ensure_safe_directory(target.parent)
            downloads.append((item, target))
        self._download_files(downloads)

    def prune_run_checkpoints(self, *, run_id: str, checkpoint_id: str) -> dict[str, object]:
        """Delete superseded checkpoint objects after latest.json points at current."""

        run_id = safe_path_component(run_id, label="run_id")
        checkpoint_id = safe_path_component(checkpoint_id, label="checkpoint_id")
        root = f"run/{run_id}/checkpoints/"
        current = f"{root}{checkpoint_id}/"
        files = self._list_files(prefix=root)
        delete = sorted(
            str(getattr(item, "path"))
            for item in files
            if str(getattr(item, "path")).startswith(root)
            and not str(getattr(item, "path")).startswith(current)
        )
        best_path = f"run/{run_id}/best.json"
        if any(getattr(item, "path", None) == best_path for item in self._list_files(prefix=best_path)):
            delete.append(best_path)
        if delete:
            self._require_method("batch_bucket_files")(
                bucket_id=self.bucket_id,
                delete=delete,
                **self._kwargs(),
            )
        pointer = self.read_json(f"run/{run_id}/latest.json")
        if not isinstance(pointer, Mapping) or pointer.get("checkpoint_id") != checkpoint_id:
            raise RuntimeError("latest checkpoint pointer changed during Hugging Face bucket cleanup")
        return {
            "status": "pruned",
            "checkpoint_id": checkpoint_id,
            "deleted_files": len(delete),
            "bucket_id": self.bucket_id,
        }


__all__ = ["HuggingFaceBucketCheckpointStore"]
