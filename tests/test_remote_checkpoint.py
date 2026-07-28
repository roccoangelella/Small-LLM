"""Offline durability, two-phase publication, and migration coverage."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from dataset.src.joint_checkpoint import CheckpointCoordinator, restore_on_empty_vps
from dataset.src.remote import (
    GoogleDriveShardStore, HuggingFaceCheckpointStore, InMemoryDriveStore,
    InMemoryHuggingFaceStore, TwoPhaseCheckpointPublisher, build_checkpoint_manifest,
    mirror_finalized_shard, sha256_bytes, write_drive_manifest,
)


class MockTrainer:
    def __init__(self) -> None:
        self.state = {"model": 0, "optimizer": 0, "global_optimizer_step": 0}
        self.load_calls = 0
    def state_dict(self): return dict(self.state)
    def load_state_dict(self, state):
        self.load_calls += 1
        self.state = dict(state)


class UploadResponseStore(InMemoryHuggingFaceStore):
    def __init__(self, mode: str | None = None) -> None:
        super().__init__()
        self.mode = mode
        self.upload_calls = 0

    def upload_tree(self, remote_prefix: str, local_dir: Path) -> dict[str, str]:
        self.upload_calls += 1
        response = super().upload_tree(remote_prefix, local_dir)
        if self.mode == "mismatch" or (self.mode == "best_mismatch" and remote_prefix.endswith("/best")):
            key = sorted(response)[0]
            response[key] = "0" * 64
        elif self.mode == "missing":
            response.pop(sorted(response)[0])
        elif self.mode == "unexpected":
            response[remote_prefix + "/unexpected.bin"] = "0" * 64
        return response


class FakeDriveFiles:
    def __init__(self, metadata: dict[str, object]) -> None:
        self.metadata = metadata

    def get(self, **kwargs):
        return SimpleNamespace(execute=lambda: dict(self.metadata))


class FakeDriveHttp:
    def __init__(self, data: bytes, *, status: int = 206, include_content_range: bool = True) -> None:
        self.data = data
        self.status = status
        self.include_content_range = include_content_range
        self.ranges: list[str] = []

    def request(self, uri: str, *, method: str, headers: dict[str, str]):
        range_header = headers.get("Range", "")
        self.ranges.append(range_header)
        if not range_header.startswith("bytes=") or "-" not in range_header[6:]:
            raise AssertionError("download must use bounded byte ranges")
        start_text, end_text = range_header[6:].split("-", 1)
        if not end_text:
            raise AssertionError("download must never use an open-ended range")
        start, end = int(start_text), int(end_text)
        if start < 0 or end < start or end >= len(self.data):
            raise AssertionError("invalid requested range")
        content = self.data[start:end + 1]
        headers = {"Content-Length": str(len(content))}
        if self.include_content_range:
            headers["Content-Range"] = f"bytes {start}-{end}/{len(self.data)}"
        response = SimpleNamespace(
            status=self.status,
            headers=headers,
        )
        return response, content


class FakeDriveService:
    def __init__(self, data: bytes, *, logical_name: str, response_status: int = 206,
                 include_content_range: bool = True) -> None:
        self._http = FakeDriveHttp(data, status=response_status, include_content_range=include_content_range)
        self._metadata = {
            "id": "file-id", "size": str(len(data)),
            "appProperties": {"logical_name": logical_name, "sha256": sha256_bytes(data)},
        }
        self._files = FakeDriveFiles(self._metadata)

    def files(self):
        return self._files


class FakeHubApi:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def upload_file(self, *, path_or_fileobj, path_in_repo: str, **kwargs):
        if isinstance(path_or_fileobj, (str, Path)):
            data = Path(path_or_fileobj).read_bytes()
        else:
            data = path_or_fileobj.read()
        # Deliberately transform uploaded bytes so the test proves the store
        # hashes a read-back Hub object rather than trusting its local digest.
        self.objects[path_in_repo] = data[::-1]


class FakeDriveListingFiles:
    def __init__(self, entries: list[dict[str, object]]) -> None:
        self.entries = entries
        self.last_query: str | None = None

    def list(self, *, q: str, fields: str):
        self.last_query = q
        return SimpleNamespace(execute=lambda: {"files": list(self.entries)})


class FakeDriveListingService:
    def __init__(self, entries: list[dict[str, object]]) -> None:
        self.file_resource = FakeDriveListingFiles(entries)

    def files(self):
        return self.file_resource


class RemoteCheckpointTest(unittest.TestCase):
    def _drive_manifest(self, root: Path, drive: InMemoryDriveStore) -> dict:
        shard = root / "cache" / "train" / "train-000000.bin"
        shard.parent.mkdir(parents=True)
        shard.write_bytes(b"\x01\x00\x02\x00")
        entry = mirror_finalized_shard(drive, run_id="run", cache_root=root,
                                      entry={"filename": "train/train-000000.bin", "byte_size": 4,
                                             "checksum": "c69b1a"}, config_hash="cfg", schema_hash="schema")
        # mirror validates a real digest, so retain the computed manifest value.
        return write_drive_manifest(root / "drive_manifest.json", run_id="run", entries=[entry],
                                    configuration_hash="cfg", schema_hash="schema")

    def test_resumable_download_and_checksum_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / "a.bin"; source.write_bytes(b"abcdef")
            drive = InMemoryDriveStore()
            uploaded = drive.upload_finalized_shard(run_id="r", logical_name="a.bin", local_path=source)
            target = root / "out" / "a.bin"; target.parent.mkdir()
            target.with_name("a.bin.part").write_bytes(b"abc")
            drive.download_shard(run_id="r", logical_name="a.bin", file_id=uploaded["file_id"], destination=target,
                                 byte_size=6, sha256=uploaded["sha256"])
            self.assertEqual(target.read_bytes(), b"abcdef")
            with self.assertRaises(RuntimeError):
                drive.verify_remote_shard(run_id="r", logical_name="a.bin", file_id=uploaded["file_id"],
                                          byte_size=6, sha256="0" * 64)

    def test_checkpoint_failure_keeps_previous_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); drive = InMemoryDriveStore()
            # Use a correct local SHA-256 while exercising immutable verification.
            shard = root / "cache" / "train" / "train-000000.bin"; shard.parent.mkdir(parents=True); shard.write_bytes(b"1234")
            import hashlib
            entry = mirror_finalized_shard(drive, run_id="run", cache_root=root / "cache",
                entry={"filename": "train/train-000000.bin", "byte_size": 4,
                       "checksum": hashlib.sha256(b"1234").hexdigest()}, config_hash="cfg", schema_hash="schema")
            manifest = write_drive_manifest(root / "drive_manifest.json", run_id="run", entries=[entry], configuration_hash="cfg", schema_hash="schema")
            trainer = MockTrainer(); coordinator = CheckpointCoordinator(root / "checkpoints", configuration_hash="cfg", source_hash="src", schema_hash="schema")
            first = coordinator.save(checkpoint_id="one", trainer=trainer,
                pipeline_state={"gradient_accumulation_position": 0, "last_consumed_block_id": 2}, optimizer_step_complete=True)
            hf = InMemoryHuggingFaceStore(); publisher = TwoPhaseCheckpointPublisher(hf, run_id="run")
            coordinator.publish(publisher, checkpoint_id="one", drive_manifest=manifest)
            previous = hf.read_json("run/run/latest.json")
            failing = InMemoryHuggingFaceStore(fail=lambda stage: (_ for _ in ()).throw(RuntimeError(stage)) if stage == "hf_pointer" else None)
            failing.objects.update(hf.objects)
            with self.assertRaises(RuntimeError):
                TwoPhaseCheckpointPublisher(failing, run_id="run").publish(first, checkpoint_id="two", drive_manifest=manifest)
            self.assertEqual(failing.read_json("run/run/latest.json"), previous)

    def test_empty_vps_migration_restores_state_and_prefetch(self) -> None:
        with tempfile.TemporaryDirectory() as first_tmp, tempfile.TemporaryDirectory() as second_tmp:
            first, second = Path(first_tmp), Path(second_tmp); drive = InMemoryDriveStore()
            import hashlib
            shard = first / "cache" / "train" / "train-000000.bin"; shard.parent.mkdir(parents=True); shard.write_bytes(b"abcd")
            entry = mirror_finalized_shard(drive, run_id="run", cache_root=first / "cache",
                entry={"filename": "train/train-000000.bin", "byte_size": 4, "checksum": hashlib.sha256(b"abcd").hexdigest()}, config_hash="cfg", schema_hash="schema")
            manifest = write_drive_manifest(first / "drive_manifest.json", run_id="run", entries=[entry], configuration_hash="cfg", schema_hash="schema")
            trainer = MockTrainer(); trainer.state["model"] = 7
            coordinator = CheckpointCoordinator(first / "checkpoints", configuration_hash="cfg", source_hash="src", schema_hash="schema")
            coordinator.save(checkpoint_id="one", trainer=trainer, pipeline_state={"gradient_accumulation_position": 0, "last_consumed_block_id": 0}, optimizer_step_complete=True)
            hf = InMemoryHuggingFaceStore(); publisher = TwoPhaseCheckpointPublisher(hf, run_id="run")
            published = coordinator.publish(publisher, checkpoint_id="one", drive_manifest=manifest)
            restore_on_empty_vps(publisher=publisher, store=drive, run_id="run", destination=second,
                                 checkpoint_pointer=published["latest"], prefetch_shards=1)
            restored = MockTrainer(); remote_coordinator = CheckpointCoordinator(second / "checkpoints", configuration_hash="cfg", source_hash="src", schema_hash="schema")
            pipeline = remote_coordinator.load("one", restored)
            self.assertEqual({key: restored.state[key] for key in trainer.state}, trainer.state)
            self.assertEqual(pipeline["last_consumed_block_id"], 0)
            self.assertTrue((second / "cache" / "train" / "train-000000.bin").exists())

    def test_shard_download_rejects_destination_parent_and_part_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outside = root / "outside"
            outside.mkdir()
            (outside / "secret.bin").write_bytes(b"secret")
            source = root / "source.bin"
            source.write_bytes(b"abcdef")

            drive = InMemoryDriveStore()
            uploaded = drive.upload_finalized_shard(
                run_id="run", logical_name="a.bin", local_path=source
            )
            intermediate = root / "escape"
            intermediate.symlink_to(outside, target_is_directory=True)
            with self.assertRaises(RuntimeError):
                drive.download_shard(
                    run_id="run", logical_name="a.bin", file_id=uploaded["file_id"],
                    destination=intermediate / "a.bin", byte_size=6, sha256=uploaded["sha256"],
                )

            destination = root / "destination.bin"
            destination.symlink_to(outside / "secret.bin")
            with self.assertRaises(RuntimeError):
                drive.download_shard(
                    run_id="run", logical_name="a.bin", file_id=uploaded["file_id"],
                    destination=destination, byte_size=6, sha256=uploaded["sha256"],
                )

            destination.unlink()
            part = destination.with_name(destination.name + ".part")
            part.symlink_to(outside / "secret.bin")
            with self.assertRaises(RuntimeError):
                drive.download_shard(
                    run_id="run", logical_name="a.bin", file_id=uploaded["file_id"],
                    destination=destination, byte_size=6, sha256=uploaded["sha256"],
                )

            google = GoogleDriveShardStore(FakeDriveService(b"abcdef", logical_name="a.bin"), "folder")
            with self.assertRaises(RuntimeError):
                google.download_shard(
                    run_id="run", logical_name="a.bin", file_id="file-id",
                    destination=intermediate / "google.bin", byte_size=6, sha256=sha256_bytes(b"abcdef"),
                )

    def test_restore_rejects_symlinked_cache_and_cleans_staging(self) -> None:
        with tempfile.TemporaryDirectory() as first_tmp, tempfile.TemporaryDirectory() as second_tmp:
            first, second = Path(first_tmp), Path(second_tmp)
            drive = InMemoryDriveStore()
            shard = first / "cache" / "train" / "train-000000.bin"
            shard.parent.mkdir(parents=True)
            shard.write_bytes(b"abcd")
            entry = mirror_finalized_shard(
                drive, run_id="run", cache_root=first / "cache",
                entry={"filename": "train/train-000000.bin", "byte_size": 4,
                       "checksum": sha256_bytes(b"abcd")},
                config_hash="cfg", schema_hash="schema",
            )
            manifest = write_drive_manifest(
                first / "drive_manifest.json", run_id="run", entries=[entry],
                configuration_hash="cfg", schema_hash="schema",
            )
            coordinator = CheckpointCoordinator(
                first / "checkpoints", configuration_hash="cfg", source_hash="src", schema_hash="schema"
            )
            coordinator.save(
                checkpoint_id="one", trainer=MockTrainer(),
                pipeline_state={"gradient_accumulation_position": 0},
                optimizer_step_complete=True,
            )
            publisher = TwoPhaseCheckpointPublisher(InMemoryHuggingFaceStore(), run_id="run")
            pointer = coordinator.publish(publisher, checkpoint_id="one", drive_manifest=manifest)["latest"]

            outside = second / "outside"
            outside.mkdir()
            cache = second / "cache"
            cache.symlink_to(outside, target_is_directory=True)
            with self.assertRaises(RuntimeError):
                restore_on_empty_vps(
                    publisher=publisher, store=drive, run_id="run", destination=second,
                    checkpoint_pointer=pointer, prefetch_shards=1,
                )
            checkpoints = second / "checkpoints"
            self.assertFalse((checkpoints / "one").exists())
            self.assertEqual(list(checkpoints.iterdir()), [])
            self.assertFalse((outside / "train" / "train-000000.bin").exists())

    def test_restore_rejects_preexisting_final_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as first_tmp, tempfile.TemporaryDirectory() as second_tmp:
            first, second = Path(first_tmp), Path(second_tmp)
            coordinator = CheckpointCoordinator(
                first / "checkpoints", configuration_hash="cfg", source_hash="src", schema_hash="schema"
            )
            coordinator.save(
                checkpoint_id="one", trainer=MockTrainer(),
                pipeline_state={"gradient_accumulation_position": 0},
                optimizer_step_complete=True,
            )
            publisher = TwoPhaseCheckpointPublisher(InMemoryHuggingFaceStore(), run_id="run")
            pointer = coordinator.publish(
                publisher, checkpoint_id="one", drive_manifest={"version": 1, "run_id": "run", "shards": []}
            )["latest"]
            final = second / "checkpoints" / "one"
            final.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                restore_on_empty_vps(
                    publisher=publisher, store=InMemoryDriveStore(), run_id="run",
                    destination=second, checkpoint_pointer=pointer, prefetch_shards=0,
                )
            self.assertTrue(final.is_dir())
            self.assertEqual(list((second / "checkpoints").iterdir()), [final])

    def test_path_components_are_rejected_before_checkpoint_or_fake_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            coordinator = CheckpointCoordinator(
                root / "checkpoints", configuration_hash="cfg", source_hash="src", schema_hash="schema"
            )
            checkpoint = coordinator.save(
                checkpoint_id="one", trainer=MockTrainer(),
                pipeline_state={"gradient_accumulation_position": 0},
                optimizer_step_complete=True,
            )
            with self.assertRaises(RuntimeError):
                coordinator.publish(
                    TwoPhaseCheckpointPublisher(InMemoryHuggingFaceStore(), run_id="run"),
                    checkpoint_id="../escape", drive_manifest={"version": 1, "shards": []},
                )
            with self.assertRaises(RuntimeError):
                TwoPhaseCheckpointPublisher(InMemoryHuggingFaceStore(), run_id="../escape")

            fake = InMemoryHuggingFaceStore()
            with self.assertRaises(RuntimeError):
                fake.upload_tree("run/../escape", checkpoint)
            with self.assertRaises(RuntimeError):
                fake.write_json("run/../pointer.json", {})

            source = root / "source.bin"
            source.write_bytes(b"x")
            drive = InMemoryDriveStore()
            with self.assertRaises(RuntimeError):
                drive.upload_finalized_shard(
                    run_id="../escape", logical_name="a.bin", local_path=source
                )

    def test_symlinked_checkpoint_tree_is_rejected_before_upload_or_exfiltration(self) -> None:
        if not hasattr(os, "symlink"):
            self.skipTest("symlinks are unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            coordinator = CheckpointCoordinator(
                root / "checkpoints", configuration_hash="cfg", source_hash="src", schema_hash="schema"
            )
            checkpoint = coordinator.save(
                checkpoint_id="one", trainer=MockTrainer(),
                pipeline_state={"gradient_accumulation_position": 0},
                optimizer_step_complete=True,
            )
            secret = root / "secret.bin"
            secret.write_bytes(b"do not upload")
            (checkpoint / "linked-secret.bin").symlink_to(secret)
            secret_dir = root / "secret-dir"
            secret_dir.mkdir()
            (secret_dir / "payload.bin").write_bytes(b"also secret")
            (checkpoint / "linked-directory").symlink_to(secret_dir, target_is_directory=True)

            with self.assertRaises(RuntimeError):
                build_checkpoint_manifest(checkpoint)
            store = InMemoryHuggingFaceStore()
            with self.assertRaises(RuntimeError):
                store.upload_tree("run/one", checkpoint)
            with self.assertRaises(RuntimeError):
                TwoPhaseCheckpointPublisher(store, run_id="run").publish(
                    checkpoint, checkpoint_id="one", drive_manifest={"version": 1, "shards": []}
                )
            self.assertEqual(store.objects, {})

    def test_corrupted_trainer_pickle_is_rejected_before_load_state_dict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            coordinator = CheckpointCoordinator(
                root / "checkpoints", configuration_hash="cfg", source_hash="src", schema_hash="schema"
            )
            coordinator.save(
                checkpoint_id="one", trainer=MockTrainer(),
                pipeline_state={"gradient_accumulation_position": 0},
                optimizer_step_complete=True,
            )
            pickle_path = root / "checkpoints" / "one" / "trainer_state.pkl"
            pickle_path.write_bytes(b"not a pickle")
            trainer = MockTrainer()
            with self.assertRaises(RuntimeError):
                coordinator.load("one", trainer)
            self.assertEqual(trainer.load_calls, 0)

    def test_downloaded_checkpoint_tree_is_manifest_verified_before_drive_manifest_read(self) -> None:
        with tempfile.TemporaryDirectory() as first_tmp, tempfile.TemporaryDirectory() as second_tmp:
            first, second = Path(first_tmp), Path(second_tmp)
            drive = InMemoryDriveStore()
            manifest = {"version": 1, "run_id": "run", "configuration_hash": "cfg",
                        "schema_hash": "schema", "shards": []}
            coordinator = CheckpointCoordinator(
                first / "checkpoints", configuration_hash="cfg", source_hash="src", schema_hash="schema"
            )
            coordinator.save(
                checkpoint_id="one", trainer=MockTrainer(),
                pipeline_state={"gradient_accumulation_position": 0},
                optimizer_step_complete=True,
            )
            hf = InMemoryHuggingFaceStore()
            publisher = TwoPhaseCheckpointPublisher(hf, run_id="run")
            published = coordinator.publish(publisher, checkpoint_id="one", drive_manifest=manifest)
            key = "run/run/checkpoints/one/last/drive_manifest.json"
            hf.objects[key] = b"{}"
            with self.assertRaises(RuntimeError):
                restore_on_empty_vps(
                    publisher=publisher, store=drive, run_id="run", destination=second,
                    checkpoint_pointer=published["latest"], prefetch_shards=0,
                )
            checkpoints = second / "checkpoints"
            self.assertFalse((checkpoints / "one").exists())
            self.assertEqual(list(checkpoints.iterdir()), [])

    def test_publisher_rejects_mismatched_missing_and_unexpected_upload_hashes(self) -> None:
        for mode in ("mismatch", "missing", "unexpected"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                coordinator = CheckpointCoordinator(
                    root / "checkpoints", configuration_hash="cfg", source_hash="src", schema_hash="schema"
                )
                checkpoint = coordinator.save(
                    checkpoint_id="one", trainer=MockTrainer(),
                    pipeline_state={"gradient_accumulation_position": 0},
                    optimizer_step_complete=True,
                )
                store = UploadResponseStore(mode)
                publisher = TwoPhaseCheckpointPublisher(store, run_id="run")
                with self.assertRaises(RuntimeError):
                    publisher.publish(checkpoint, checkpoint_id="one",
                                     drive_manifest={"version": 1, "run_id": "run", "shards": []})
                self.assertNotIn("run/run/latest.json", store.objects)

    def test_publisher_rejects_wrong_run_and_malformed_shard_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            coordinator = CheckpointCoordinator(
                root / "checkpoints", configuration_hash="cfg", source_hash="src", schema_hash="schema"
            )
            checkpoint = coordinator.save(
                checkpoint_id="one", trainer=MockTrainer(),
                pipeline_state={"gradient_accumulation_position": 0},
                optimizer_step_complete=True,
            )
            store = InMemoryHuggingFaceStore()
            publisher = TwoPhaseCheckpointPublisher(store, run_id="run")
            with self.assertRaises(RuntimeError):
                publisher.publish(
                    checkpoint, checkpoint_id="one",
                    drive_manifest={"version": 1, "run_id": "other", "shards": []},
                )
            with self.assertRaises(RuntimeError):
                publisher.publish(
                    checkpoint, checkpoint_id="one",
                    drive_manifest={"version": 1, "run_id": "run", "shards": ["bad"]},
                )
            self.assertNotIn("run/run/latest.json", store.objects)

    def test_best_upload_hashes_are_verified_before_best_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            coordinator = CheckpointCoordinator(
                root / "checkpoints", configuration_hash="cfg", source_hash="src", schema_hash="schema"
            )
            checkpoint = coordinator.save(
                checkpoint_id="one", trainer=MockTrainer(),
                pipeline_state={"gradient_accumulation_position": 0},
                optimizer_step_complete=True,
            )
            store = UploadResponseStore("best_mismatch")
            publisher = TwoPhaseCheckpointPublisher(store, run_id="run")
            with self.assertRaises(RuntimeError):
                publisher.publish(checkpoint, checkpoint_id="one",
                                 drive_manifest={"version": 1, "run_id": "run", "shards": []},
                                 metric=2.0, best_metric=1.0)
            self.assertIn("run/run/latest.json", store.objects)
            self.assertNotIn("run/run/best.json", store.objects)

    def test_google_drive_download_uses_bounded_resumable_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = b"x" * (GoogleDriveShardStore.DOWNLOAD_RANGE_SIZE + 17)
            service = FakeDriveService(data, logical_name="a.bin")
            store = GoogleDriveShardStore(service, folder_id="folder")
            destination = root / "out" / "a.bin"
            destination.parent.mkdir()
            partial = destination.with_name("a.bin.part")
            partial.write_bytes(data[:3])
            store.download_shard(
                run_id="run", logical_name="a.bin", file_id="file-id", destination=destination,
                byte_size=len(data), sha256=sha256_bytes(data),
            )
            self.assertEqual(destination.read_bytes(), data)
            self.assertFalse(partial.exists())
            self.assertGreaterEqual(len(service._http.ranges), 2)
            for value in service._http.ranges:
                self.assertRegex(value, r"^bytes=\d+-\d+$")
                start, end = (int(part) for part in value[6:].split("-"))
                self.assertLessEqual(end - start + 1, GoogleDriveShardStore.DOWNLOAD_RANGE_SIZE)

            complete_destination = root / "out" / "complete.bin"
            complete_part = complete_destination.with_name("complete.bin.part")
            complete_part.write_bytes(data)
            requests_before = len(service._http.ranges)
            store.download_shard(
                run_id="run", logical_name="a.bin", file_id="file-id", destination=complete_destination,
                byte_size=len(data), sha256=sha256_bytes(data),
            )
            self.assertEqual(len(service._http.ranges), requests_before)
            self.assertEqual(complete_destination.read_bytes(), data)

    def test_google_drive_requires_a_matching_206_content_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = b"abcdef"
            for status, include_content_range in ((200, True), (206, False)):
                with self.subTest(status=status, include_content_range=include_content_range):
                    service = FakeDriveService(
                        data, logical_name="a.bin", response_status=status,
                        include_content_range=include_content_range,
                    )
                    store = GoogleDriveShardStore(service, folder_id="folder")
                    destination = Path(tmp) / f"out-{status}-{include_content_range}.bin"
                    with self.assertRaises(RuntimeError):
                        store.download_shard(
                            run_id="run", logical_name="a.bin", file_id="file-id",
                            destination=destination, byte_size=len(data), sha256=sha256_bytes(data),
                        )
                    self.assertFalse(destination.exists())

    def test_drive_query_escaping_and_duplicate_folder_rejection(self) -> None:
        service = FakeDriveListingService([{"id": "folder-id"}])
        store = GoogleDriveShardStore(service, folder_id="parent'id")
        self.assertEqual(store._run_folder("run'id"), "folder-id")
        assert service.file_resource.last_query is not None
        self.assertIn("run\\'id", service.file_resource.last_query)
        self.assertIn("parent\\'id", service.file_resource.last_query)

        duplicate_service = FakeDriveListingService([{"id": "one"}, {"id": "two"}])
        with self.assertRaises(RuntimeError):
            GoogleDriveShardStore(duplicate_service, folder_id="folder")._run_folder("run")

    def test_drive_manifest_run_id_and_shard_fields_are_validated_before_prefetch(self) -> None:
        with tempfile.TemporaryDirectory() as first_tmp, tempfile.TemporaryDirectory() as second_tmp:
            first, second = Path(first_tmp), Path(second_tmp)
            coordinator = CheckpointCoordinator(
                first / "checkpoints", configuration_hash="cfg", source_hash="src", schema_hash="schema"
            )
            coordinator.save(
                checkpoint_id="one", trainer=MockTrainer(),
                pipeline_state={"gradient_accumulation_position": 0},
                optimizer_step_complete=True,
            )
            publisher = TwoPhaseCheckpointPublisher(InMemoryHuggingFaceStore(), run_id="run")
            bad_run_manifest = {"version": 1, "run_id": "other", "shards": []}
            with self.assertRaises(RuntimeError):
                coordinator.publish(
                    publisher, checkpoint_id="one", drive_manifest=bad_run_manifest
                )
            self.assertFalse((second / "checkpoints" / "one").exists())

            second_cleanup = Path(second_tmp) / "second-clean"
            good_run_bad_shard = {
                "version": 1, "run_id": "run",
                "shards": [{
                    "filename": "train/a.bin", "drive_file_id": "id", "byte_size": True,
                    "local_sha256": "0" * 64, "remote_durable": True,
                }],
            }
            checkpoint_two = coordinator.save(
                checkpoint_id="two", trainer=MockTrainer(),
                pipeline_state={"gradient_accumulation_position": 0},
                optimizer_step_complete=True,
            )
            pointer_two = publisher.publish(
                checkpoint_two, checkpoint_id="two", drive_manifest=good_run_bad_shard
            )["latest"]
            with self.assertRaises(RuntimeError):
                restore_on_empty_vps(
                    publisher=publisher, store=InMemoryDriveStore(), run_id="run",
                    destination=second_cleanup, checkpoint_pointer=pointer_two, prefetch_shards=1,
                )
            self.assertFalse((second_cleanup / "checkpoints" / "two").exists())

    def test_huggingface_upload_hashes_read_back_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local = root / "tree"
            local.mkdir()
            source = local / "payload.bin"
            source.write_bytes(b"uploaded bytes")
            api = FakeHubApi()
            download_root = root / "downloads"

            def downloader(*, filename: str, **kwargs) -> str:
                target = download_root / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(api.objects[filename])
                return str(target)

            store = HuggingFaceCheckpointStore(
                "org/private", token="token", revision="main", private=True,
                api=api, downloader=downloader,
            )
            hashes = store.upload_tree("run/one", local)
            key = "run/one/payload.bin"
            self.assertEqual(hashes[key], sha256_bytes(b"uploaded bytes"[::-1]))
            self.assertNotEqual(hashes[key], sha256_bytes(source.read_bytes()))


if __name__ == "__main__":
    unittest.main()
