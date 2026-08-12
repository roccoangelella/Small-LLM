import hashlib, json, tempfile, unittest
from pathlib import Path
from trainer import SchemaV2ShardReader
from tests.trainer_fixtures import payload


class RecordingCache:
    def __init__(self) -> None:
        self.ensure_calls = []
        self.ack_calls = []
        self.restore_calls = []

    def ensure_block(self, block_id):
        self.ensure_calls.append(block_id)

    def acknowledge(self, block_id):
        self.ack_calls.append(block_id)

    def restore_after_acknowledged(self, block_id):
        self.restore_calls.append(block_id)


class ShardReaderTests(unittest.TestCase):
    def test_resume_checksum_and_empty_vps_manifest(self):
        sequences = [[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7],[5,6,7,8]]
        raw = payload(sequences)
        checksum = hashlib.sha256(raw).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "train").mkdir()
            path = root / "train/train-000000.bin"; path.write_bytes(raw)
            shard = {"filename":"train/train-000000.bin","split":"train",
                "byte_size":len(raw),"sequence_count":5,"checksum":checksum,
                "first_block_id":0,"last_block_id":2}
            manifest = {"schema_version":2,"sequence_format":"context_plus_one",
                "context_length":3,"stored_tokens_per_sequence":4,
                "sequences_per_block":2,"shards":[shard]}
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            reader = SchemaV2ShardReader(root, semantic_vocab_size=16)
            self.assertEqual(reader.next_batch().sequence_count, 2)
            reader.acknowledge(0); state = reader.pipeline_state()
            restored = SchemaV2ShardReader(root, semantic_vocab_size=16)
            restored.load_pipeline_state(state)
            self.assertEqual(restored.next_batch().block_id, 1)
            restored.acknowledge(1)
            self.assertEqual(restored.next_batch().sequence_count, 1)

            checkpoint = root / "checkpoints/step-1"; checkpoint.mkdir(parents=True)
            cache = root / "restored-cache"; (cache / "train").mkdir(parents=True)
            (cache / "train/train-000000.bin").write_bytes(raw)
            drive = {"version":1,"run_id":"test","configuration_hash":"a"*64,
                "schema_hash":"b"*64,"shards":[{**shard,"local_sha256":checksum,
                "drive_file_id":"file-id","remote_durable":True}]}
            (checkpoint / "drive_manifest.json").write_text(json.dumps(drive), encoding="utf-8")
            migrated = SchemaV2ShardReader.from_restored_checkpoint(checkpoint, cache,
                context_length=3, sequences_per_block=2, semantic_vocab_size=16)
            migrated.load_pipeline_state(state)
            self.assertEqual(migrated.next_batch().block_id, 1)
            migrated = SchemaV2ShardReader.from_restored_checkpoint(checkpoint, cache,
                context_length=3, sequences_per_block=2, semantic_vocab_size=16)
            migrated.load_pipeline_state({"version":1,"last_consumed_block_id":0,
                "gradient_accumulation_position":0,
                "consumer":{"kind":"live_schema_v2","split":"train",
                            "last_consumed_block_id":0}})
            self.assertEqual(migrated.next_batch().block_id, 1)
            path.write_bytes(raw[:-2] + b"\x00\x00")
            with self.assertRaises(RuntimeError):
                SchemaV2ShardReader(root, semantic_vocab_size=16).next_batch()

    def test_reader_calls_rolling_cache_before_read_after_ack_and_on_restore(self):
        sequences = [[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7]]
        raw = payload(sequences)
        checksum = hashlib.sha256(raw).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "train").mkdir()
            (root / "train/train-000000.bin").write_bytes(raw)
            manifest = {"schema_version":2,"sequence_format":"context_plus_one",
                "context_length":3,"stored_tokens_per_sequence":4,
                "sequences_per_block":2,"shards":[{
                    "filename":"train/train-000000.bin","split":"train",
                    "byte_size":len(raw),"sequence_count":4,"checksum":checksum,
                    "first_block_id":0,"last_block_id":1}]}
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            cache = RecordingCache()
            reader = SchemaV2ShardReader(root, semantic_vocab_size=16, cache_manager=cache)
            batch = reader.next_batch()
            self.assertEqual(batch.block_id, 0)
            self.assertEqual(cache.ensure_calls, [0])
            reader.acknowledge(0)
            self.assertEqual(cache.ack_calls, [0])
            state = reader.pipeline_state()

            restored_cache = RecordingCache()
            restored = SchemaV2ShardReader(root, semantic_vocab_size=16,
                                            cache_manager=restored_cache)
            restored.load_pipeline_state(state)
            self.assertEqual(restored_cache.restore_calls, [0])
            self.assertEqual(restored.next_batch().block_id, 1)
            self.assertEqual(restored_cache.ensure_calls, [1])


if __name__ == "__main__":
    unittest.main()
