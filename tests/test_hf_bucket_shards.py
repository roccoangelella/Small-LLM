"""Offline tests for immutable dataset shards in Hugging Face Storage Buckets."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from dataset.src.hf_bucket_shards import HuggingFaceBucketShardStore
from dataset.src.remote import sha256_bytes


class FakeBucketApi:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.created: list[tuple[str, bool, bool, str | None]] = []

    def create_bucket(self, *, bucket_id: str, private: bool, exist_ok: bool, token=None):
        self.created.append((bucket_id, private, exist_ok, token))
        return SimpleNamespace(bucket_id=bucket_id)

    def batch_bucket_files(self, *, bucket_id: str, add=None, delete=None, token=None):
        del bucket_id, token
        for source, destination in add or []:
            data = source if isinstance(source, bytes) else Path(source).read_bytes()
            self.objects[destination] = bytes(data)
        for path in delete or []:
            self.objects.pop(path, None)

    def list_bucket_tree(self, *, bucket_id: str, prefix=None, recursive=True, token=None):
        del bucket_id, recursive, token
        prefix = "" if prefix is None else prefix
        return [
            SimpleNamespace(type="file", path=path, size=len(data))
            for path, data in sorted(self.objects.items())
            if path.startswith(prefix)
        ]

    def download_bucket_files(
        self,
        *,
        bucket_id: str,
        files,
        raise_on_missing_files: bool,
        token=None,
    ):
        del bucket_id, token
        for source, destination in files:
            path = source.path if hasattr(source, "path") else str(source)
            data = self.objects.get(path)
            if data is None:
                if raise_on_missing_files:
                    raise FileNotFoundError(path)
                continue
            target = Path(destination)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)


class HuggingFaceBucketShardStoreTests(unittest.TestCase):
    def _store(self, api: FakeBucketApi) -> HuggingFaceBucketShardStore:
        return HuggingFaceBucketShardStore(
            "owner/datasets",
            token="token",
            private=True,
            api=api,
            create_bucket=True,
        )

    def test_upload_is_read_back_verified_and_download_is_atomic(self) -> None:
        api = FakeBucketApi()
        store = self._store(api)
        self.assertEqual(api.created, [("owner/datasets", True, True, "token")])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "train-000000.bin"
            source.write_bytes(b"immutable-shard")
            digest = sha256_bytes(source.read_bytes())
            uploaded = store.upload_finalized_shard(
                run_id="dataset-001",
                logical_name="train/train-000000.bin",
                local_path=source,
            )
            self.assertEqual(uploaded["sha256"], digest)
            verified = store.verify_remote_shard(
                run_id="dataset-001",
                logical_name="train/train-000000.bin",
                file_id="run/dataset-001/train/train-000000.bin",
                byte_size=source.stat().st_size,
                sha256=digest,
            )
            self.assertEqual(verified["sha256"], digest)
            destination = root / "cache" / "train" / "train-000000.bin"
            store.download_shard(
                run_id="dataset-001",
                logical_name="train/train-000000.bin",
                file_id="run/dataset-001/train/train-000000.bin",
                destination=destination,
                byte_size=source.stat().st_size,
                sha256=digest,
            )
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            self.assertFalse(destination.with_name(destination.name + ".part").exists())

    def test_existing_object_cannot_be_replaced_with_different_bytes(self) -> None:
        api = FakeBucketApi()
        store = self._store(api)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "train-000000.bin"
            source.write_bytes(b"first")
            store.upload_finalized_shard(
                run_id="dataset-001",
                logical_name="train/train-000000.bin",
                local_path=source,
            )
            source.write_bytes(b"other")
            with self.assertRaises(RuntimeError):
                store.upload_finalized_shard(
                    run_id="dataset-001",
                    logical_name="train/train-000000.bin",
                    local_path=source,
                )

    def test_manifest_ready_pointer_is_published_last_and_verified(self) -> None:
        api = FakeBucketApi()
        store = self._store(api)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {
                "production": {"run_id": "dataset-001", "target_reached": True},
                "shards": [
                    {"filename": "train/train-000000.bin", "split": "train"},
                    {"filename": "validation/validation-000000.bin", "split": "validation"},
                ],
            }
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            ready = store.publish_dataset_manifest(run_id="dataset-001", manifest_path=path)
            self.assertTrue(ready["target_reached"])
            self.assertIn("run/dataset-001/manifest.json", api.objects)
            self.assertIn("run/dataset-001/ready.json", api.objects)
            restored = root / "restored-manifest.json"
            downloaded = store.download_dataset_manifest(
                run_id="dataset-001",
                destination=restored,
            )
            self.assertEqual(downloaded, manifest)


if __name__ == "__main__":
    unittest.main()
