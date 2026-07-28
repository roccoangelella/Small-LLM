"""Offline durability, two-phase publication, and migration coverage."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dataset.src.joint_checkpoint import CheckpointCoordinator, restore_on_empty_vps
from dataset.src.remote import (
    InMemoryDriveStore, InMemoryHuggingFaceStore, TwoPhaseCheckpointPublisher,
    mirror_finalized_shard, write_drive_manifest,
)


class MockTrainer:
    def __init__(self) -> None:
        self.state = {"model": 0, "optimizer": 0, "global_optimizer_step": 0}
    def state_dict(self): return dict(self.state)
    def load_state_dict(self, state): self.state = dict(state)


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


if __name__ == "__main__":
    unittest.main()
