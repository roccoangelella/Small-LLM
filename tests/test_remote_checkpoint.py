"""Offline durability, two-phase publication, and legacy-manifest migration coverage."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from dataset.src.joint_checkpoint import CheckpointCoordinator, restore_on_empty_vps
from dataset.src.remote import (
    HuggingFaceCheckpointStore,
    InMemoryHuggingFaceStore,
    InMemoryShardStore,
    TwoPhaseCheckpointPublisher,
    build_checkpoint_manifest,
    mirror_finalized_shard,
    sha256_bytes,
    write_drive_manifest,
)


class MockTrainer:
    def __init__(self) -> None:
        self.state = {"model": 0, "optimizer": 0, "global_optimizer_step": 0}
        self.load_calls = 0

    def state_dict(self):
        return dict(self.state)

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
        if self.mode == "mismatch" or (
            self.mode == "best_mismatch" and remote_prefix.endswith("/best")
        ):
            key = sorted(response)[0]
            response[key] = "0" * 64
        elif self.mode == "missing":
            response.pop(sorted(response)[0])
        elif self.mode == "unexpected":
            response[remote_prefix + "/unexpected.bin"] = "0" * 64
        return response


class FakeHubApi:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def upload_file(self, *, path_or_fileobj, path_in_repo: str, **kwargs):
        del kwargs
        if isinstance(path_or_fileobj, (str, Path)):
            data = Path(path_or_fileobj).read_bytes()
        else:
            data = path_or_fileobj.read()
        # Transform the uploaded bytes so the test proves upload_tree returns
        # the digest of a remote read-back rather than blindly trusting local bytes.
        self.objects[path_in_repo] = data[::-1]


class RemoteCheckpointTest(unittest.TestCase):
    @staticmethod
    def _one_shard_manifest(
        root: Path,
        store: InMemoryShardStore,
        *,
        first_block_id: int = 0,
        last_block_id: int = 0,
    ) -> dict[str, object]:
        shard = root / "cache" / "train" / "train-000000.bin"
        shard.parent.mkdir(parents=True, exist_ok=True)
        shard.write_bytes(b"abcd")
        entry = mirror_finalized_shard(
            store,
            run_id="run",
            cache_root=root / "cache",
            entry={
                "filename": "train/train-000000.bin",
                "split": "train",
                "byte_size": 4,
                "checksum": sha256_bytes(b"abcd"),
                "first_block_id": first_block_id,
                "last_block_id": last_block_id,
            },
            config_hash="cfg",
            schema_hash="schema",
        )
        return write_drive_manifest(
            root / "drive_manifest.json",
            run_id="run",
            entries=[entry],
            configuration_hash="cfg",
            schema_hash="schema",
        )

    def test_resumable_download_and_checksum_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "a.bin"
            source.write_bytes(b"abcdef")
            store = InMemoryShardStore()
            uploaded = store.upload_finalized_shard(
                run_id="r", logical_name="a.bin", local_path=source
            )
            target = root / "out" / "a.bin"
            target.parent.mkdir()
            target.with_name("a.bin.part").write_bytes(b"abc")
            store.download_shard(
                run_id="r",
                logical_name="a.bin",
                file_id=str(uploaded["file_id"]),
                destination=target,
                byte_size=6,
                sha256=str(uploaded["sha256"]),
            )
            self.assertEqual(target.read_bytes(), b"abcdef")
            with self.assertRaises(RuntimeError):
                store.verify_remote_shard(
                    run_id="r",
                    logical_name="a.bin",
                    file_id=str(uploaded["file_id"]),
                    byte_size=6,
                    sha256="0" * 64,
                )

    def test_checkpoint_failure_keeps_previous_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shards = InMemoryShardStore()
            manifest = self._one_shard_manifest(root, shards)
            trainer = MockTrainer()
            coordinator = CheckpointCoordinator(
                root / "checkpoints",
                configuration_hash="cfg",
                source_hash="src",
                schema_hash="schema",
            )
            first = coordinator.save(
                checkpoint_id="one",
                trainer=trainer,
                pipeline_state={
                    "gradient_accumulation_position": 0,
                    "last_consumed_block_id": 0,
                },
                optimizer_step_complete=True,
            )
            hf = InMemoryHuggingFaceStore()
            publisher = TwoPhaseCheckpointPublisher(hf, run_id="run")
            coordinator.publish(publisher, checkpoint_id="one", drive_manifest=manifest)
            previous = hf.read_json("run/run/latest.json")
            failing = InMemoryHuggingFaceStore(
                fail=lambda stage: (
                    (_ for _ in ()).throw(RuntimeError(stage))
                    if stage == "hf_pointer"
                    else None
                )
            )
            failing.objects.update(hf.objects)
            with self.assertRaises(RuntimeError):
                TwoPhaseCheckpointPublisher(failing, run_id="run").publish(
                    first,
                    checkpoint_id="two",
                    drive_manifest=manifest,
                )
            self.assertEqual(failing.read_json("run/run/latest.json"), previous)

    def test_empty_host_migration_restores_state_and_prefetch(self) -> None:
        with tempfile.TemporaryDirectory() as first_tmp, tempfile.TemporaryDirectory() as second_tmp:
            first, second = Path(first_tmp), Path(second_tmp)
            shards = InMemoryShardStore()
            manifest = self._one_shard_manifest(first, shards)
            trainer = MockTrainer()
            trainer.state["model"] = 7
            coordinator = CheckpointCoordinator(
                first / "checkpoints",
                configuration_hash="cfg",
                source_hash="src",
                schema_hash="schema",
            )
            coordinator.save(
                checkpoint_id="one",
                trainer=trainer,
                pipeline_state={
                    "gradient_accumulation_position": 0,
                    "last_consumed_block_id": -1,
                },
                optimizer_step_complete=True,
            )
            hf = InMemoryHuggingFaceStore()
            publisher = TwoPhaseCheckpointPublisher(hf, run_id="run")
            published = coordinator.publish(
                publisher,
                checkpoint_id="one",
                drive_manifest=manifest,
            )
            restore_on_empty_vps(
                publisher=publisher,
                store=shards,
                run_id="run",
                destination=second,
                checkpoint_pointer=published["latest"],
                prefetch_shards=1,
            )
            restored = MockTrainer()
            remote_coordinator = CheckpointCoordinator(
                second / "checkpoints",
                configuration_hash="cfg",
                source_hash="src",
                schema_hash="schema",
            )
            pipeline = remote_coordinator.load("one", restored)
            self.assertEqual(restored.state, trainer.state)
            self.assertEqual(pipeline["last_consumed_block_id"], -1)
            self.assertTrue((second / "cache" / "train" / "train-000000.bin").exists())

    def test_prefetch_starts_at_shard_containing_next_unconsumed_block(self) -> None:
        with tempfile.TemporaryDirectory() as first_tmp, tempfile.TemporaryDirectory() as second_tmp:
            first, second = Path(first_tmp), Path(second_tmp)
            shards = InMemoryShardStore()
            entries = []
            for index, (first_block, last_block) in enumerate(((0, 1), (2, 3), (4, 5))):
                filename = f"train/train-{index:06d}.bin"
                shard = first / "cache" / filename
                shard.parent.mkdir(parents=True, exist_ok=True)
                shard.write_bytes(bytes([index + 1]) * 4)
                entries.append(
                    mirror_finalized_shard(
                        shards,
                        run_id="run",
                        cache_root=first / "cache",
                        entry={
                            "filename": filename,
                            "split": "train",
                            "byte_size": 4,
                            "checksum": sha256_bytes(shard.read_bytes()),
                            "first_block_id": first_block,
                            "last_block_id": last_block,
                        },
                        config_hash="cfg",
                        schema_hash="schema",
                    )
                )
            manifest = write_drive_manifest(
                first / "drive_manifest.json",
                run_id="run",
                entries=entries,
                configuration_hash="cfg",
                schema_hash="schema",
            )
            coordinator = CheckpointCoordinator(
                first / "checkpoints",
                configuration_hash="cfg",
                source_hash="src",
                schema_hash="schema",
            )
            coordinator.save(
                checkpoint_id="one",
                trainer=MockTrainer(),
                pipeline_state={
                    "gradient_accumulation_position": 0,
                    "last_consumed_block_id": 2,
                },
                optimizer_step_complete=True,
            )
            publisher = TwoPhaseCheckpointPublisher(InMemoryHuggingFaceStore(), run_id="run")
            pointer = coordinator.publish(
                publisher,
                checkpoint_id="one",
                drive_manifest=manifest,
            )["latest"]
            restore_on_empty_vps(
                publisher=publisher,
                store=shards,
                run_id="run",
                destination=second,
                checkpoint_pointer=pointer,
                prefetch_shards=2,
            )
            self.assertFalse((second / "cache" / "train" / "train-000000.bin").exists())
            self.assertTrue((second / "cache" / "train" / "train-000001.bin").exists())
            self.assertTrue((second / "cache" / "train" / "train-000002.bin").exists())

    def test_shard_download_rejects_destination_parent_and_part_symlinks(self) -> None:
        if not hasattr(os, "symlink"):
            self.skipTest("symlinks are unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outside = root / "outside"
            outside.mkdir()
            (outside / "secret.bin").write_bytes(b"secret")
            source = root / "source.bin"
            source.write_bytes(b"abcdef")
            store = InMemoryShardStore()
            uploaded = store.upload_finalized_shard(
                run_id="run", logical_name="a.bin", local_path=source
            )

            intermediate = root / "escape"
            intermediate.symlink_to(outside, target_is_directory=True)
            with self.assertRaises(RuntimeError):
                store.download_shard(
                    run_id="run",
                    logical_name="a.bin",
                    file_id=str(uploaded["file_id"]),
                    destination=intermediate / "a.bin",
                    byte_size=6,
                    sha256=str(uploaded["sha256"]),
                )

            destination = root / "destination.bin"
            destination.symlink_to(outside / "secret.bin")
            with self.assertRaises(RuntimeError):
                store.download_shard(
                    run_id="run",
                    logical_name="a.bin",
                    file_id=str(uploaded["file_id"]),
                    destination=destination,
                    byte_size=6,
                    sha256=str(uploaded["sha256"]),
                )

            destination.unlink()
            part = destination.with_name(destination.name + ".part")
            part.symlink_to(outside / "secret.bin")
            with self.assertRaises(RuntimeError):
                store.download_shard(
                    run_id="run",
                    logical_name="a.bin",
                    file_id=str(uploaded["file_id"]),
                    destination=destination,
                    byte_size=6,
                    sha256=str(uploaded["sha256"]),
                )

    def test_restore_rejects_symlinked_cache_and_cleans_staging(self) -> None:
        if not hasattr(os, "symlink"):
            self.skipTest("symlinks are unavailable")
        with tempfile.TemporaryDirectory() as first_tmp, tempfile.TemporaryDirectory() as second_tmp:
            first, second = Path(first_tmp), Path(second_tmp)
            shards = InMemoryShardStore()
            manifest = self._one_shard_manifest(first, shards)
            coordinator = CheckpointCoordinator(
                first / "checkpoints",
                configuration_hash="cfg",
                source_hash="src",
                schema_hash="schema",
            )
            coordinator.save(
                checkpoint_id="one",
                trainer=MockTrainer(),
                pipeline_state={
                    "gradient_accumulation_position": 0,
                    "last_consumed_block_id": -1,
                },
                optimizer_step_complete=True,
            )
            publisher = TwoPhaseCheckpointPublisher(InMemoryHuggingFaceStore(), run_id="run")
            pointer = coordinator.publish(
                publisher,
                checkpoint_id="one",
                drive_manifest=manifest,
            )["latest"]

            outside = second / "outside"
            outside.mkdir()
            cache = second / "cache"
            cache.symlink_to(outside, target_is_directory=True)
            with self.assertRaises(RuntimeError):
                restore_on_empty_vps(
                    publisher=publisher,
                    store=shards,
                    run_id="run",
                    destination=second,
                    checkpoint_pointer=pointer,
                    prefetch_shards=1,
                )
            checkpoints = second / "checkpoints"
            self.assertFalse((checkpoints / "one").exists())
            self.assertEqual(list(checkpoints.iterdir()), [])
            self.assertFalse((outside / "train" / "train-000000.bin").exists())

    def test_restore_rejects_preexisting_final_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as first_tmp, tempfile.TemporaryDirectory() as second_tmp:
            first, second = Path(first_tmp), Path(second_tmp)
            coordinator = CheckpointCoordinator(
                first / "checkpoints",
                configuration_hash="cfg",
                source_hash="src",
                schema_hash="schema",
            )
            coordinator.save(
                checkpoint_id="one",
                trainer=MockTrainer(),
                pipeline_state={"gradient_accumulation_position": 0},
                optimizer_step_complete=True,
            )
            publisher = TwoPhaseCheckpointPublisher(InMemoryHuggingFaceStore(), run_id="run")
            pointer = coordinator.publish(
                publisher,
                checkpoint_id="one",
                drive_manifest={"version": 1, "run_id": "run", "shards": []},
            )["latest"]
            final = second / "checkpoints" / "one"
            final.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                restore_on_empty_vps(
                    publisher=publisher,
                    store=InMemoryShardStore(),
                    run_id="run",
                    destination=second,
                    checkpoint_pointer=pointer,
                    prefetch_shards=0,
                )
            self.assertTrue(final.is_dir())
            self.assertEqual(list((second / "checkpoints").iterdir()), [final])

    def test_path_components_are_rejected_before_checkpoint_or_fake_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            coordinator = CheckpointCoordinator(
                root / "checkpoints",
                configuration_hash="cfg",
                source_hash="src",
                schema_hash="schema",
            )
            checkpoint = coordinator.save(
                checkpoint_id="one",
                trainer=MockTrainer(),
                pipeline_state={"gradient_accumulation_position": 0},
                optimizer_step_complete=True,
            )
            with self.assertRaises(RuntimeError):
                coordinator.publish(
                    TwoPhaseCheckpointPublisher(InMemoryHuggingFaceStore(), run_id="run"),
                    checkpoint_id="../escape",
                    drive_manifest={"version": 1, "shards": []},
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
            shards = InMemoryShardStore()
            with self.assertRaises(RuntimeError):
                shards.upload_finalized_shard(
                    run_id="../escape",
                    logical_name="a.bin",
                    local_path=source,
                )

    def test_symlinked_checkpoint_tree_is_rejected_before_upload_or_exfiltration(self) -> None:
        if not hasattr(os, "symlink"):
            self.skipTest("symlinks are unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            coordinator = CheckpointCoordinator(
                root / "checkpoints",
                configuration_hash="cfg",
                source_hash="src",
                schema_hash="schema",
            )
            checkpoint = coordinator.save(
                checkpoint_id="one",
                trainer=MockTrainer(),
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
                    checkpoint,
                    checkpoint_id="one",
                    drive_manifest={"version": 1, "run_id": "run", "shards": []},
                )
            self.assertEqual(store.objects, {})

    def test_corrupted_trainer_pickle_is_rejected_before_load_state_dict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            coordinator = CheckpointCoordinator(
                root / "checkpoints",
                configuration_hash="cfg",
                source_hash="src",
                schema_hash="schema",
            )
            coordinator.save(
                checkpoint_id="one",
                trainer=MockTrainer(),
                pipeline_state={"gradient_accumulation_position": 0},
                optimizer_step_complete=True,
            )
            pickle_path = root / "checkpoints" / "one" / "trainer_state.pkl"
            pickle_path.write_bytes(b"not a pickle")
            trainer = MockTrainer()
            with self.assertRaises(RuntimeError):
                coordinator.load("one", trainer)
            self.assertEqual(trainer.load_calls, 0)

    def test_downloaded_checkpoint_tree_is_manifest_verified_before_legacy_manifest_read(self) -> None:
        with tempfile.TemporaryDirectory() as first_tmp, tempfile.TemporaryDirectory() as second_tmp:
            first, second = Path(first_tmp), Path(second_tmp)
            shards = InMemoryShardStore()
            manifest = {
                "version": 1,
                "run_id": "run",
                "configuration_hash": "cfg",
                "schema_hash": "schema",
                "shards": [],
            }
            coordinator = CheckpointCoordinator(
                first / "checkpoints",
                configuration_hash="cfg",
                source_hash="src",
                schema_hash="schema",
            )
            coordinator.save(
                checkpoint_id="one",
                trainer=MockTrainer(),
                pipeline_state={"gradient_accumulation_position": 0},
                optimizer_step_complete=True,
            )
            hf = InMemoryHuggingFaceStore()
            publisher = TwoPhaseCheckpointPublisher(hf, run_id="run")
            published = coordinator.publish(
                publisher,
                checkpoint_id="one",
                drive_manifest=manifest,
            )
            key = "run/run/checkpoints/one/last/drive_manifest.json"
            hf.objects[key] = b"{}"
            with self.assertRaises(RuntimeError):
                restore_on_empty_vps(
                    publisher=publisher,
                    store=shards,
                    run_id="run",
                    destination=second,
                    checkpoint_pointer=published["latest"],
                    prefetch_shards=0,
                )
            checkpoints = second / "checkpoints"
            self.assertFalse((checkpoints / "one").exists())
            self.assertEqual(list(checkpoints.iterdir()), [])

    def test_publisher_rejects_mismatched_missing_and_unexpected_upload_hashes(self) -> None:
        for mode in ("mismatch", "missing", "unexpected"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                coordinator = CheckpointCoordinator(
                    root / "checkpoints",
                    configuration_hash="cfg",
                    source_hash="src",
                    schema_hash="schema",
                )
                checkpoint = coordinator.save(
                    checkpoint_id="one",
                    trainer=MockTrainer(),
                    pipeline_state={"gradient_accumulation_position": 0},
                    optimizer_step_complete=True,
                )
                store = UploadResponseStore(mode)
                publisher = TwoPhaseCheckpointPublisher(store, run_id="run")
                with self.assertRaises(RuntimeError):
                    publisher.publish(
                        checkpoint,
                        checkpoint_id="one",
                        drive_manifest={"version": 1, "run_id": "run", "shards": []},
                    )
                self.assertNotIn("run/run/latest.json", store.objects)

    def test_publisher_rejects_wrong_run_and_malformed_shard_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            coordinator = CheckpointCoordinator(
                root / "checkpoints",
                configuration_hash="cfg",
                source_hash="src",
                schema_hash="schema",
            )
            checkpoint = coordinator.save(
                checkpoint_id="one",
                trainer=MockTrainer(),
                pipeline_state={"gradient_accumulation_position": 0},
                optimizer_step_complete=True,
            )
            store = InMemoryHuggingFaceStore()
            publisher = TwoPhaseCheckpointPublisher(store, run_id="run")
            with self.assertRaises(RuntimeError):
                publisher.publish(
                    checkpoint,
                    checkpoint_id="one",
                    drive_manifest={"version": 1, "run_id": "other", "shards": []},
                )
            with self.assertRaises(RuntimeError):
                publisher.publish(
                    checkpoint,
                    checkpoint_id="one",
                    drive_manifest={"version": 1, "run_id": "run", "shards": ["bad"]},
                )
            self.assertNotIn("run/run/latest.json", store.objects)

    def test_best_upload_hashes_are_verified_before_best_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            coordinator = CheckpointCoordinator(
                root / "checkpoints",
                configuration_hash="cfg",
                source_hash="src",
                schema_hash="schema",
            )
            checkpoint = coordinator.save(
                checkpoint_id="one",
                trainer=MockTrainer(),
                pipeline_state={"gradient_accumulation_position": 0},
                optimizer_step_complete=True,
            )
            store = UploadResponseStore("best_mismatch")
            publisher = TwoPhaseCheckpointPublisher(store, run_id="run")
            with self.assertRaises(RuntimeError):
                publisher.publish(
                    checkpoint,
                    checkpoint_id="one",
                    drive_manifest={"version": 1, "run_id": "run", "shards": []},
                    metric=2.0,
                    best_metric=1.0,
                )
            self.assertIn("run/run/latest.json", store.objects)
            self.assertNotIn("run/run/best.json", store.objects)

    def test_legacy_manifest_run_id_and_shard_fields_are_validated_before_prefetch(self) -> None:
        with tempfile.TemporaryDirectory() as first_tmp, tempfile.TemporaryDirectory() as second_tmp:
            first, second = Path(first_tmp), Path(second_tmp)
            coordinator = CheckpointCoordinator(
                first / "checkpoints",
                configuration_hash="cfg",
                source_hash="src",
                schema_hash="schema",
            )
            coordinator.save(
                checkpoint_id="one",
                trainer=MockTrainer(),
                pipeline_state={"gradient_accumulation_position": 0},
                optimizer_step_complete=True,
            )
            publisher = TwoPhaseCheckpointPublisher(InMemoryHuggingFaceStore(), run_id="run")
            with self.assertRaises(RuntimeError):
                coordinator.publish(
                    publisher,
                    checkpoint_id="one",
                    drive_manifest={"version": 1, "run_id": "other", "shards": []},
                )

            bad_shard_manifest = {
                "version": 1,
                "run_id": "run",
                "shards": [
                    {
                        "filename": "train/a.bin",
                        "drive_file_id": "id",
                        "byte_size": True,
                        "local_sha256": "0" * 64,
                        "remote_durable": True,
                    }
                ],
            }
            checkpoint_two = coordinator.save(
                checkpoint_id="two",
                trainer=MockTrainer(),
                pipeline_state={"gradient_accumulation_position": 0},
                optimizer_step_complete=True,
            )
            pointer_two = publisher.publish(
                checkpoint_two,
                checkpoint_id="two",
                drive_manifest=bad_shard_manifest,
            )["latest"]
            with self.assertRaises(RuntimeError):
                restore_on_empty_vps(
                    publisher=publisher,
                    store=InMemoryShardStore(),
                    run_id="run",
                    destination=second,
                    checkpoint_pointer=pointer_two,
                    prefetch_shards=1,
                )
            self.assertFalse((second / "checkpoints" / "two").exists())

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
                del kwargs
                target = download_root / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(api.objects[filename])
                return str(target)

            store = HuggingFaceCheckpointStore(
                "org/private",
                token="token",
                revision="main",
                private=True,
                api=api,
                downloader=downloader,
            )
            hashes = store.upload_tree("run/one", local)
            key = "run/one/payload.bin"
            self.assertEqual(hashes[key], sha256_bytes(b"uploaded bytes"[::-1]))
            self.assertNotEqual(hashes[key], sha256_bytes(source.read_bytes()))


if __name__ == "__main__":
    unittest.main()
