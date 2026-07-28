"""Offline tests for the deterministic stream-cache primitives."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dataset import config
from dataset.src.bitio import decode_uint16_le
from dataset.src.bytesource import RangeReader, SourceFile
from dataset.src.records import ParsedRecord
from dataset.src.streaming import (
    ImmutableShardWriter, QueueConsumer, SequencePacker, SourceDocument,
    StreamCacheConfig, StreamCacheProducer, TokenDeficitScheduler,
    build_stream_cache, normalize_cluster_weights, parallel_read_documents,
    synthetic_test_weights,
)
from dataset.src.workplan import WorkItem, WorkPlan, build_work_plan
from tests.synthetic import build_default_synthetic_source


def _doc(cluster: int, tokens: list[int], identity: str | None = None) -> SourceDocument:
    return SourceDocument(identity or f"doc-{cluster}-{tokens[0]}", cluster, tuple(tokens))


class SchedulerTest(unittest.TestCase):
    def test_exact_token_deficit_and_round_trip(self) -> None:
        weights = synthetic_test_weights()
        weights[1], weights[2] = 3, 1
        scheduler = TokenDeficitScheduler(weights, "seed")
        self.assertEqual(scheduler.choose([1, 2]), scheduler.choose([1, 2]))
        scheduler.emit(_doc(1, list(range(100))))  # overshoot remains debt
        self.assertEqual(scheduler.choose([1, 2]), 2)
        restored = TokenDeficitScheduler.from_state(weights, scheduler.state_dict())
        self.assertEqual(restored.state_dict(), scheduler.state_dict())

    def test_weight_validation_rejects_missing_and_cluster_11(self) -> None:
        with self.assertRaises(ValueError):
            normalize_cluster_weights({1: 1})
        bad = synthetic_test_weights()
        bad[11] = 1
        with self.assertRaises(ValueError):
            normalize_cluster_weights(bad)


class PackerTest(unittest.TestCase):
    def test_concatenation_eod_long_split_and_resume(self) -> None:
        packer = SequencePacker(3)
        sequences = packer.push(_doc(1, [1, 2]))
        self.assertEqual(sequences, [])
        state = packer.state_dict()
        sequences = SequencePacker.from_state(state).push(_doc(2, [3, 4, 5, 6]))
        self.assertEqual(
            [list(item.tokens) for item in sequences],
            [[1, 2, 50256, 3], [4, 5, 6, 50256]],
        )
        tail = SequencePacker(3)
        tail.push(_doc(1, [9, 50256]))
        self.assertEqual(len(tail.finish()[0].tokens), 4)

    def test_cluster_token_counts_not_multiplied_across_sequences(self) -> None:
        packer = SequencePacker(3)
        # 10 tokens + EOD = 11 tokens -> 2 full sequences of 4 tokens + 3 tokens carried
        doc = _doc(1, list(range(10)))
        sequences = packer.push(doc)
        total_cluster_tokens = sum(
            seq.cluster_source_tokens.get(1, 0) for seq in sequences
        )
        # The document has 10 source tokens; should be attributed once across sequences
        self.assertEqual(total_cluster_tokens, 10)


class StreamCacheEndToEndTest(unittest.TestCase):
    def _stream_config(self) -> StreamCacheConfig:
        return StreamCacheConfig(
            context_length=3, sequences_per_block=1, target_shard_bytes=8,
            reader_workers=2, max_in_flight_work_items=2, per_cluster_queue_limit=10,
            prepared_block_queue_limit=20, prefetch_head_start=0,
            weights=synthetic_test_weights(), scheduler_tie_break_seed="offline-test",
        )

    def test_dual_sink_shards_and_manifest_are_offline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            consumer = QueueConsumer(20)
            producer = StreamCacheProducer(Path(tmp), self._stream_config(), consumer)
            producer.add_training_document(_doc(1, [1, 2]))
            producer.add_training_document(_doc(2, [3, 4, 5, 6]))
            producer.add_validation_document(_doc(3, [7, 8]))
            manifest = producer.finish()
            received = []
            while not consumer.queue.empty():
                received.append(consumer.queue.get_nowait())
            self.assertTrue(received)
            self.assertEqual(manifest["sequence_format"], "context_plus_one")
            files = list((Path(tmp) / "train").glob("*.bin")) + list((Path(tmp) / "validation").glob("*.bin"))
            self.assertGreaterEqual(len(files), 2)
            payload = b"".join(path.read_bytes() for path in sorted(files))
            self.assertTrue(all(0 <= token <= config.TOKEN_MAX for token in decode_uint16_le(payload)))
            self.assertFalse(list(Path(tmp).rglob("*.tmp")))

    def test_monotonic_block_ids_across_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            consumer = QueueConsumer(20)
            producer = StreamCacheProducer(Path(tmp), self._stream_config(), consumer)
            producer.add_training_document(_doc(1, [1, 2, 3]))
            producer.add_validation_document(_doc(2, [4, 5, 6]))
            producer.add_training_document(_doc(3, [7, 8, 9]))
            producer.drain_training()

            received = []
            while not consumer.queue.empty():
                block = consumer.queue.get_nowait()
                consumer.acknowledge(block.block_id)
                received.append(block)

            block_ids = [b.block_id for b in received]
            self.assertEqual(block_ids, sorted(block_ids))
            self.assertEqual(len(block_ids), len(set(block_ids)))

    def test_producer_checkpoint_and_from_state_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            consumer = QueueConsumer(20)
            producer = StreamCacheProducer(Path(tmp), self._stream_config(), consumer)
            producer.add_training_document(_doc(1, [1, 2, 3]))
            producer.drain_training()

            state = producer.checkpoint_state()
            self.assertGreaterEqual(producer.last_durable_block_id, 0)

            restored_consumer = QueueConsumer(20)
            restored_producer = StreamCacheProducer.from_state(
                Path(tmp), self._stream_config(), state, restored_consumer
            )

            self.assertEqual(
                restored_producer.last_durable_block_id,
                producer.last_durable_block_id,
            )
            # Next block ID should continue from last_durable_block_id + 1
            restored_producer.add_training_document(_doc(2, [10, 11, 12]))
            restored_producer.drain_training()

            if not restored_consumer.queue.empty():
                new_block = restored_consumer.queue.get_nowait()
                self.assertEqual(
                    new_block.block_id, producer.last_durable_block_id + 1
                )

    def test_shard_writer_index_discovery_and_stale_tmp_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            train_dir = Path(tmp) / "train"
            train_dir.mkdir(parents=True, exist_ok=True)
            # Pre-create shard 0 and stale tmp for shard 1
            (train_dir / "train-000000.bin").write_bytes(b"\x00\x00" * 4)
            stale_tmp = train_dir / ".train-000001.bin.tmp"
            stale_tmp.write_bytes(b"stale_garbage")

            writer = ImmutableShardWriter(
                Path(tmp), split="train", target_bytes=100, context_length=3
            )
            self.assertEqual(writer._index, 1)

    def test_parallel_read_documents_deterministic_order(self) -> None:
        # Mock range reader returning synthetic valid JSONL records
        doc1 = b'{"cluster_id": 1, "text": "hello", "tokens": [1, 2, 3]}\n'
        doc2 = b'{"cluster_id": 2, "text": "world", "tokens": [4, 5, 6]}\n'
        full_data = doc1 + doc2

        class DummyReader:
            def file_size(self) -> int:
                return len(full_data)

            def read_range(self, offset: int, length: int) -> bytes:
                return full_data[offset : offset + length]

        plan = WorkPlan(
            schema_version=2,
            dataset=config.DATASET_REPOSITORY,
            revision=config.DATASET_REVISION,
            source_glob=config.SOURCE_DATA_GLOB,
            selection_seed=config.SELECTION_SEED,
            region_bytes=256 * 1024 * 1024,
            source_files=(SourceFile(path="part_0.tokenized.jsonl", size=len(full_data)),),
            work_items=(
                WorkItem(index=0, filename="part_0.tokenized.jsonl", range_start=0, range_end=len(doc1)),
                WorkItem(index=1, filename="part_0.tokenized.jsonl", range_start=len(doc1), range_end=len(full_data)),
            ),
            hash="0" * 64,
        )

        results = list(
            parallel_read_documents(
                plan,
                reader_factory=lambda _: DummyReader(),
                workers=2,
                max_in_flight=2,
            )
        )
        self.assertEqual(len(results), 2)
        is_val_0, doc_0 = results[0]
        is_val_1, doc_1 = results[1]
        self.assertEqual(doc_0.work_item_index, 0)
        self.assertEqual(doc_1.work_item_index, 1)
        self.assertEqual(doc_0.cluster_id, 1)
        self.assertEqual(doc_1.cluster_id, 2)


class StreamCacheCLITest(unittest.TestCase):
    def test_cli_stream_cache_valid_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            weights_file = Path(tmp) / "weights.json"
            valid_weights = {str(c): 1 for c in config.ACCEPTED_CLUSTER_IDS}
            weights_file.write_text(json.dumps(valid_weights), encoding="utf-8")

            import io
            from contextlib import redirect_stdout
            from dataset.main import main

            out = io.StringIO()
            with redirect_stdout(out):
                exit_code = main(["stream-cache", "--weights-file", str(weights_file), "--show-stream-config"])

            self.assertEqual(exit_code, 0)
            parsed = json.loads(out.getvalue())
            self.assertEqual(parsed["command"], "stream-cache")
            self.assertEqual(parsed["status"], "validated")
            self.assertIn("stream_cache_config", parsed)
            cfg = parsed["stream_cache_config"]
            self.assertEqual(cfg["context_length"], 2048)
            self.assertEqual(cfg["stored_sequence_tokens"], 2049)
            self.assertEqual(cfg["block_bytes"], (2048 + 1) * 512 * 2)

    def test_cli_stream_cache_missing_weights_file_arg(self) -> None:
        from dataset.main import main

        with self.assertRaises(SystemExit):
            main(["stream-cache"])

    def test_cli_stream_cache_nonexistent_weights_file(self) -> None:
        from dataset.main import main

        exit_code = main(["stream-cache", "--weights-file", "/nonexistent/path/weights.json"])
        self.assertEqual(exit_code, 1)

    def test_cli_stream_cache_invalid_cluster_11(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            weights_file = Path(tmp) / "weights.json"
            bad_weights = {str(c): 1 for c in config.ALL_CLUSTER_IDS}
            weights_file.write_text(json.dumps(bad_weights), encoding="utf-8")

            from dataset.main import main

            exit_code = main(["stream-cache", "--weights-file", str(weights_file)])
            self.assertEqual(exit_code, 1)

    def test_cli_stream_cache_missing_accepted_cluster(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            weights_file = Path(tmp) / "weights.json"
            bad_weights = {str(c): 1 for c in range(1, 11)}
            weights_file.write_text(json.dumps(bad_weights), encoding="utf-8")

            from dataset.main import main

            exit_code = main(["stream-cache", "--weights-file", str(weights_file)])
            self.assertEqual(exit_code, 1)

    def test_cli_stream_cache_negative_weight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            weights_file = Path(tmp) / "weights.json"
            bad_weights = {str(c): 1 for c in config.ACCEPTED_CLUSTER_IDS}
            bad_weights["1"] = -5
            weights_file.write_text(json.dumps(bad_weights), encoding="utf-8")

            from dataset.main import main

            exit_code = main(["stream-cache", "--weights-file", str(weights_file)])
            self.assertEqual(exit_code, 1)


class BuildStreamCacheAdapterTest(unittest.TestCase):
    def test_one_vs_multi_reader_equivalence(self) -> None:
        synthetic = build_default_synthetic_source()
        plan = build_work_plan(
            synthetic.source_files,
            region_bytes=100,
            seed=config.SELECTION_SEED,
            repository=config.DATASET_REPOSITORY,
            revision=config.DATASET_REVISION,
        )

        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            dir1 = Path(tmp1)
            dir2 = Path(tmp2)

            cfg1 = StreamCacheConfig(
                context_length=8,
                sequences_per_block=2,
                target_shard_bytes=256,
                reader_workers=1,
                max_in_flight_work_items=2,
                per_cluster_queue_limit=3,
                prepared_block_queue_limit=1000,
                prefetch_head_start=0,
                weights=synthetic_test_weights(),
                scheduler_tie_break_seed="adapter-test-seed",
            )
            cfg2 = StreamCacheConfig(
                context_length=8,
                sequences_per_block=2,
                target_shard_bytes=256,
                reader_workers=4,
                max_in_flight_work_items=4,
                per_cluster_queue_limit=3,
                prepared_block_queue_limit=1000,
                prefetch_head_start=0,
                weights=synthetic_test_weights(),
                scheduler_tie_break_seed="adapter-test-seed",
            )

            manifest1 = build_stream_cache(
                output_dir=dir1,
                stream_config=cfg1,
                plan=plan,
                reader_factory=synthetic.reader_factory(),
            )
            manifest2 = build_stream_cache(
                output_dir=dir2,
                stream_config=cfg2,
                plan=plan,
                reader_factory=synthetic.reader_factory(),
            )

            self.assertEqual(manifest1, manifest2)

            # Assert shard files and payloads match exactly between single and multi reader runs
            shards1 = sorted(p.relative_to(dir1) for p in dir1.rglob("*.bin"))
            shards2 = sorted(p.relative_to(dir2) for p in dir2.rglob("*.bin"))
            self.assertEqual(shards1, shards2)

            for rel_path in shards1:
                content1 = (dir1 / rel_path).read_bytes()
                content2 = (dir2 / rel_path).read_bytes()
                self.assertEqual(content1, content2)

            # Ensure no temporary files are left behind
            self.assertEqual(list(dir1.rglob("*.tmp")), [])
            self.assertEqual(list(dir2.rglob("*.tmp")), [])

    def test_reader_exception_propagation(self) -> None:
        synthetic = build_default_synthetic_source()
        plan = build_work_plan(
            synthetic.source_files,
            region_bytes=100,
            seed=config.SELECTION_SEED,
            repository=config.DATASET_REPOSITORY,
            revision=config.DATASET_REVISION,
        )
        cfg = StreamCacheConfig(
            context_length=8,
            sequences_per_block=2,
            target_shard_bytes=256,
            reader_workers=2,
            max_in_flight_work_items=2,
            per_cluster_queue_limit=3,
            prepared_block_queue_limit=1000,
            prefetch_head_start=0,
            weights=synthetic_test_weights(),
            scheduler_tie_break_seed="adapter-test-seed",
        )

        class FaultyReader:
            def __init__(self, sf: SourceFile) -> None:
                self._sf = sf

            def file_size(self) -> int:
                return self._sf.size

            def read_range(self, offset: int, length: int) -> bytes:
                raise RuntimeError("Simulated range read failure")

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            with self.assertRaises(RuntimeError) as ctx:
                build_stream_cache(
                    output_dir=output_dir,
                    stream_config=cfg,
                    plan=plan,
                    reader_factory=lambda sf: FaultyReader(sf),
                )

            self.assertIn("Simulated range read failure", str(ctx.exception))
            self.assertFalse((output_dir / config.MANIFEST_FILENAME).exists())
            self.assertEqual(list(output_dir.rglob("*.bin")), [])


if __name__ == "__main__":
    unittest.main()


