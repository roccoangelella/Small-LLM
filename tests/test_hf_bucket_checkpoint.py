"""Offline tests for the mutable Hugging Face Storage Bucket checkpoint store."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from dataset.src.hf_bucket_checkpoint import HuggingFaceBucketCheckpointStore
from dataset.src.remote import sha256_bytes


class _FakeBucketApi:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.created: list[tuple[str, bool, bool, str | None]] = []
        self.batches: list[dict[str, object]] = []

    def create_bucket(self, *, bucket_id: str, private: bool, exist_ok: bool, token=None):
        self.created.append((bucket_id, private, exist_ok, token))
        return SimpleNamespace(bucket_id=bucket_id)

    def batch_bucket_files(self, *, bucket_id: str, add=None, delete=None, token=None):
        row = {"bucket_id": bucket_id, "add": add, "delete": delete, "token": token}
        self.batches.append(row)
        for source, destination in add or []:
            data = source if isinstance(source, bytes) else Path(source).read_bytes()
            self.objects[destination] = bytes(data)
        for path in delete or []:
            self.objects.pop(path, None)
        return None

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
        return None


class HuggingFaceBucketCheckpointStoreTests(unittest.TestCase):
    def _store(self, api: _FakeBucketApi) -> HuggingFaceBucketCheckpointStore:
        return HuggingFaceBucketCheckpointStore(
            "owner/checkpoints",
            token="token",
            private=True,
            api=api,
            create_bucket=True,
        )

    def test_create_upload_readback_and_download_tree(self) -> None:
        api = _FakeBucketApi()
        store = self._store(api)
        self.assertEqual(api.created, [("owner/checkpoints", True, True, "token")])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "checkpoint.json").write_bytes(b"checkpoint")
            (source / "nested").mkdir()
            (source / "nested" / "state.bin").write_bytes(b"state")

            uploaded = store.upload_tree("run/r/checkpoints/step-1/last", source)
            self.assertEqual(
                uploaded,
                {
                    "run/r/checkpoints/step-1/last/checkpoint.json": sha256_bytes(b"checkpoint"),
                    "run/r/checkpoints/step-1/last/nested/state.bin": sha256_bytes(b"state"),
                },
            )

            destination = root / "restore"
            store.download_tree("run/r/checkpoints/step-1/last", destination)
            self.assertEqual((destination / "checkpoint.json").read_bytes(), b"checkpoint")
            self.assertEqual((destination / "nested" / "state.bin").read_bytes(), b"state")

    def test_json_pointer_is_read_back(self) -> None:
        api = _FakeBucketApi()
        store = self._store(api)
        pointer = {"checkpoint_id": "step-1", "value": 7}
        store.write_json("run/r/latest.json", pointer)
        self.assertEqual(store.read_json("run/r/latest.json"), pointer)
        self.assertIsNone(store.read_json("run/r/missing.json"))

    def test_prune_keeps_current_checkpoint_and_latest_pointer(self) -> None:
        api = _FakeBucketApi()
        store = self._store(api)
        api.objects.update(
            {
                "run/r/checkpoints/step-1/last/a.bin": b"old",
                "run/r/checkpoints/step-1/last/checkpoint_manifest.json": b"old-manifest",
                "run/r/checkpoints/step-2/last/a.bin": b"new",
                "run/r/checkpoints/step-2/last/checkpoint_manifest.json": b"new-manifest",
                "run/r/best.json": b'{"checkpoint_id":"step-1"}',
            }
        )
        store.write_json("run/r/latest.json", {"checkpoint_id": "step-2"})

        result = store.prune_run_checkpoints(run_id="r", checkpoint_id="step-2")
        self.assertEqual(result["status"], "pruned")
        self.assertNotIn("run/r/checkpoints/step-1/last/a.bin", api.objects)
        self.assertNotIn("run/r/best.json", api.objects)
        self.assertEqual(api.objects["run/r/checkpoints/step-2/last/a.bin"], b"new")
        self.assertEqual(store.read_json("run/r/latest.json"), {"checkpoint_id": "step-2"})

    def test_prune_fails_closed_if_latest_pointer_is_not_current(self) -> None:
        api = _FakeBucketApi()
        store = self._store(api)
        api.objects["run/r/checkpoints/step-2/last/a.bin"] = b"new"
        store.write_json("run/r/latest.json", {"checkpoint_id": "step-1"})
        with self.assertRaisesRegex(RuntimeError, "latest checkpoint pointer changed"):
            store.prune_run_checkpoints(run_id="r", checkpoint_id="step-2")


if __name__ == "__main__":
    unittest.main()
