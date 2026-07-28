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
    ImmutableShardWriter, PackedSequence, PreparedBlockBuilder, QueueConsumer,
    SequencePacker, SourceDocument, StreamCacheConfig, StreamCacheProducer,
    TokenDeficitScheduler,
    build_stream_cache, normalize_cluster_weights, parallel_read_document_batches, parallel_read_documents,
    synthetic_test_weights,
)
from dataset.src.workplan import WorkItem, WorkPlan, build_work_plan
from tests.synthetic import build_default_synthetic_source


def _doc(cluster: int, tokens: list[int], identity: str | None = None) -> SourceDocument:
    return SourceDocument(identity or f"doc-{cluster}-{tokens[0]}", cluster, tuple(tokens))


class _AcknowledgingQueueConsumer(QueueConsumer):
    def submit(self, block) -> None:
        super().submit(block)
        self.acknowledge(block.block_id)


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
            [[1, 2, 50256, 3], [3, 4, 5, 6]],
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

    def test_every_next_token_transition_is_trained_once(self) -> None:
        packer = SequencePacker(4)
        sequences = packer.push(_doc(1, [10, 11, 12]))
        sequences += packer.push(_doc(2, [20, 21, 22, 23, 24]))
        sequences += packer.finish()
        trained = [(sequence.tokens[index], sequence.tokens[index + 1])
                   for sequence in sequences for index in range(4)
                   if sequence.token_kinds[index + 1] != "padding"]
        original = [10, 11, 12, config.EOD_TOKEN_ID, 20, 21, 22, 23, 24, config.EOD_TOKEN_ID]
        self.assertEqual(trained, list(zip(original, original[1:])))
        self.assertEqual(sequences[0].tokens[-1], sequences[1].tokens[0])
        self.assertEqual(sum(sum(s.cluster_source_tokens.values()) for s in sequences), 8)

    def test_overlap_state_round_trip_is_exact(self) -> None:
        packer = SequencePacker(2)
        self.assertEqual([list(s.tokens) for s in packer.push(_doc(1, [1, 2, 3]))], [[1, 2, 3]])
        restored = SequencePacker.from_state(packer.state_dict())
        second = restored.push(_doc(2, [4]))
        self.assertEqual([list(s.tokens) for s in second], [[3, 50256, 4]])
        self.assertEqual(second[0].token_kinds[0], "overlap_source")

    def test_prepared_block_builder_pending_sequence_resume(self) -> None:
        sequence_one = PackedSequence(
            (1, 2, 3, 4), "first", "first", {1: 3},
            ("source", "source", "source", "inserted_eod"),
            (1, 1, 1, None),
        )
        sequence_two = PackedSequence(
            (5, 6, 7, 8), "second", "second", {2: 3},
            ("source", "source", "source", "inserted_eod"),
            (2, 2, 2, None),
        )
        original = PreparedBlockBuilder(2)
        self.assertIsNone(original.push(sequence_one, split="train", cumulative_source_tokens=3))
        restored = PreparedBlockBuilder.from_state(original.state_dict())
        resumed = restored.push(sequence_two, split="train", cumulative_source_tokens=6)

        fresh = PreparedBlockBuilder(2)
        fresh.push(sequence_one, split="train", cumulative_source_tokens=3)
        expected = fresh.push(sequence_two, split="train", cumulative_source_tokens=6)
        self.assertIsNotNone(resumed)
        self.assertEqual(resumed, expected)
        self.assertEqual(restored.state_dict()["next_block_id"], 1)


class StreamCacheEndToEndTest(unittest.TestCase):
    def _stream_config(self) -> StreamCacheConfig:
        return StreamCacheConfig(
            context_length=3, sequences_per_block=1, target_shard_bytes=8,
            reader_workers=2, max_in_flight_work_items=2, per_cluster_queue_limit=10,
            prepared_block_queue_limit=20, prefetch_head_start=0,
            minimum_prefetched_source_tokens=0, minimum_populated_cluster_queues=1,
            weights=synthetic_test_weights(), scheduler_tie_break_seed="offline-test",
        )

    def test_dual_sink_shards_and_manifest_are_offline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            consumer = _AcknowledgingQueueConsumer(20)
            validation_consumer = _AcknowledgingQueueConsumer(20)
            producer = StreamCacheProducer(
                Path(tmp), self._stream_config(), consumer,
                validation_consumer=validation_consumer,
            )
            producer.add_training_document(_doc(1, [1, 2]))
            producer.add_training_document(_doc(2, [3, 4, 5, 6]))
            producer.add_validation_document(_doc(3, [7, 8]))
            manifest = producer.finish()
            received = []
            while not consumer.queue.empty():
                received.append(consumer.queue.get_nowait())
            validation_received = []
            while not validation_consumer.queue.empty():
                validation_received.append(validation_consumer.queue.get_nowait())
            self.assertTrue(received)
            self.assertTrue(validation_received)
            self.assertTrue(all(block.split == "train" for block in received))
            self.assertTrue(all(block.split == "validation" for block in validation_received))
            self.assertEqual(manifest["sequence_format"], "context_plus_one")
            files = list((Path(tmp) / "train").glob("*.bin")) + list((Path(tmp) / "validation").glob("*.bin"))
            self.assertGreaterEqual(len(files), 2)
            payload = b"".join(path.read_bytes() for path in sorted(files))
            self.assertTrue(all(0 <= token <= config.TOKEN_MAX for token in decode_uint16_le(payload)))
            self.assertFalse(list(Path(tmp).rglob("*.tmp")))

    def test_train_and_validation_block_ids_have_separate_namespaces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            consumer = QueueConsumer(20)
            validation_consumer = QueueConsumer(20)
            producer = StreamCacheProducer(
                Path(tmp), self._stream_config(), consumer,
                validation_consumer=validation_consumer,
            )
            producer.add_training_document(_doc(1, [1, 2, 3]))
            producer.add_validation_document(_doc(2, [4, 5, 6]))
            producer.add_training_document(_doc(3, [7, 8, 9]))
            producer.drain_training()

            received = []
            while not consumer.queue.empty():
                block = consumer.queue.get_nowait()
                consumer.acknowledge(block.block_id)
                received.append(block)
            validation_received = []
            while not validation_consumer.queue.empty():
                validation_received.append(validation_consumer.queue.get_nowait())

            train_ids = [b.block_id for b in received]
            validation_ids = [b.block_id for b in validation_received]
            self.assertEqual(train_ids, [0, 1])
            self.assertEqual(validation_ids, [0])
            self.assertEqual(producer.last_durable_block_id, 1)
            self.assertEqual(producer.last_durable_validation_block_id, 0)

    def test_producer_checkpoint_and_from_state_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            consumer = QueueConsumer(20)
            producer = StreamCacheProducer(Path(tmp), self._stream_config(), consumer)
            producer.add_training_document(_doc(1, [1, 2, 3]))
            producer.drain_training()
            while not consumer.queue.empty():
                consumer.acknowledge(consumer.queue.get_nowait().block_id)

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

    def test_mixture_bound_blocks_normal_candidate_but_force_overrides(self) -> None:
        cfg = StreamCacheConfig(
            context_length=3, sequences_per_block=1, target_shard_bytes=8,
            reader_workers=1, max_in_flight_work_items=1, per_cluster_queue_limit=10,
            prepared_block_queue_limit=20, prefetch_head_start=0,
            minimum_prefetched_source_tokens=0, minimum_populated_cluster_queues=1,
            maximum_rolling_mixture_error=0.1, maximum_waiting_documents=0,
            rolling_mixture_windows=(100,), weights=synthetic_test_weights(),
            scheduler_tie_break_seed="mixture-test",
        )
        with tempfile.TemporaryDirectory() as tmp:
            producer = StreamCacheProducer(Path(tmp), cfg, QueueConsumer(20))
            producer.add_training_document(_doc(1, [1, 2, 3]))
            self.assertEqual(producer.drain_training(force=False), 1)
            producer.add_training_document(_doc(1, [4, 5, 6]))
            self.assertEqual(producer.drain_training(force=False), 0)
            self.assertEqual(producer.queued_source_tokens, 3)
            self.assertEqual(producer.drain_training(force=True), 1)
            self.assertGreater(producer.mixture_measurements()["rolling_error"], 0.1)

    def test_mixture_bootstrap_and_corrective_alternative_selection(self) -> None:
        weights = synthetic_test_weights()
        weights[1] = weights[2] = 1000
        cfg = StreamCacheConfig(
            context_length=3, sequences_per_block=1, target_shard_bytes=100_000,
            reader_workers=1, max_in_flight_work_items=1, per_cluster_queue_limit=20,
            prepared_block_queue_limit=20, prefetch_head_start=0,
            minimum_prefetched_source_tokens=0, minimum_populated_cluster_queues=1,
            maximum_rolling_mixture_error=0.1, rolling_mixture_windows=(10,),
            weights=weights, scheduler_tie_break_seed="corrective-test",
        )
        with tempfile.TemporaryDirectory() as tmp:
            producer = StreamCacheProducer(Path(tmp), cfg)
            # The first document is explicitly allowed to bootstrap the window.
            producer.add_training_document(_doc(2, [2] * 100))
            self.assertEqual(producer.drain_training(force=False), 1)
            # Build a rolling window dominated by cluster 1 while cumulative
            # deficits still make cluster 1 the scheduler's global preference.
            producer.add_training_document(_doc(1, [1] * 10))
            self.assertEqual(producer.drain_training(force=True), 1)
            producer.add_training_document(_doc(1, [3]))
            producer.add_training_document(_doc(2, [4]))
            self.assertEqual(producer.scheduler.choose([1, 2]), 1)
            self.assertEqual(producer.drain_training(force=False), 1)
            self.assertEqual(producer.scheduler.emitted_source_tokens[2], 101)

    def test_rolling_history_is_compact_exact_and_restorable(self) -> None:
        cfg = StreamCacheConfig(
            context_length=3, sequences_per_block=1, target_shard_bytes=100_000,
            reader_workers=1, max_in_flight_work_items=1, per_cluster_queue_limit=20,
            prepared_block_queue_limit=20, prefetch_head_start=0,
            minimum_prefetched_source_tokens=0, minimum_populated_cluster_queues=1,
            rolling_mixture_windows=(5,), weights=synthetic_test_weights(),
            scheduler_tie_break_seed="compact-rolling",
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            producer = StreamCacheProducer(output, cfg)
            producer.add_training_document(_doc(1, [1] * 3))
            producer.drain_training(force=False)
            producer.add_training_document(_doc(2, [2] * 4))
            producer.drain_training(force=True)
            self.assertEqual(producer.rolling_source_tokens, 5)
            self.assertEqual(
                [(item.cluster_id, item.token_count) for item in producer._rolling_contributions],
                [(1, 1), (2, 4)],
            )
            producer.add_training_document(_doc(3, [3] * 10))
            producer.drain_training(force=True)
            self.assertEqual(
                [(item.cluster_id, item.token_count) for item in producer._rolling_contributions],
                [(3, 5)],
            )
            state = producer.checkpoint_state()
            self.assertEqual(state["schema_version"], 2)
            self.assertIn("rolling_contributions", state)
            self.assertNotIn("rolling_documents", state)
            self.assertNotIn("tokens", state["rolling_contributions"][0])
            restored = StreamCacheProducer.from_state(output, cfg, state)
            self.assertEqual(
                list(restored._rolling_contributions), list(producer._rolling_contributions)
            )
            self.assertEqual(restored.rolling_source_tokens, 5)

    def test_queue_pressure_drains_the_specific_full_cluster(self) -> None:
        cfg = StreamCacheConfig(
            context_length=3, sequences_per_block=1, target_shard_bytes=100_000,
            reader_workers=1, max_in_flight_work_items=1, per_cluster_queue_limit=1,
            prepared_block_queue_limit=20, prefetch_head_start=0,
            minimum_prefetched_source_tokens=0, minimum_populated_cluster_queues=1,
            weights=synthetic_test_weights(), scheduler_tie_break_seed="queue-pressure",
        )
        with tempfile.TemporaryDirectory() as tmp:
            producer = StreamCacheProducer(Path(tmp), cfg)
            producer.add_training_document(_doc(1, [1, 2, 3]))
            producer.add_training_document(_doc(2, [4, 5, 6]))
            self.assertEqual(
                producer.drain_training(force=True, maximum_documents=1, cluster_id=1), 1
            )
            self.assertEqual(len(producer._queues[1]), 0)
            self.assertEqual(len(producer._queues[2]), 1)

    def test_checkpoint_refuses_unacknowledged_durable_consumer_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            consumer = QueueConsumer(20)
            producer = StreamCacheProducer(Path(tmp), self._stream_config(), consumer)
            producer.add_training_document(_doc(1, [1, 2, 3]))
            producer.drain_training()
            with self.assertRaises(RuntimeError):
                producer.checkpoint_state()
            block = consumer.queue.get_nowait()
            consumer.acknowledge(block.block_id)
            state = producer.checkpoint_state()
            self.assertEqual(state["last_consumer_acknowledged_block_id"], block.block_id)

    def test_checkpoint_applies_ack_invariant_to_validation_consumer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            validation_consumer = QueueConsumer(20)
            producer = StreamCacheProducer(
                Path(tmp), self._stream_config(),
                validation_consumer=validation_consumer,
            )
            producer.add_validation_document(_doc(1, [1, 2, 3]))
            with self.assertRaises(RuntimeError):
                producer.checkpoint_state()
            validation_block = validation_consumer.queue.get_nowait()
            validation_consumer.acknowledge(validation_block.block_id)
            state = producer.checkpoint_state()
            self.assertEqual(
                state["last_validation_consumer_acknowledged_block_id"],
                validation_block.block_id,
            )

    def test_mixture_measurement_uses_integer_counters_not_cumulative_token_tuples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            producer = StreamCacheProducer(Path(tmp), self._stream_config(), QueueConsumer(20))
            producer.scheduler.emitted_source_tokens[1] = 10**9
            producer.scheduler.total_emitted_source_tokens = 10**9
            measurements = producer.mixture_measurements()
            self.assertIsInstance(measurements["cumulative_error"], float)
            self.assertEqual(producer.scheduler.total_emitted_source_tokens, 10**9)
            self.assertEqual(producer.queued_source_tokens, 0)
            self.assertEqual(producer.rolling_source_tokens, 0)

    def test_checkpoint_finalizes_and_resume_manifest_keeps_previous_shards(self) -> None:
        cfg = StreamCacheConfig(
            context_length=3, sequences_per_block=1, target_shard_bytes=8,
            reader_workers=1, max_in_flight_work_items=1, per_cluster_queue_limit=10,
            prepared_block_queue_limit=20, prefetch_head_start=0,
            minimum_prefetched_source_tokens=0, minimum_populated_cluster_queues=1,
            weights=synthetic_test_weights(), scheduler_tie_break_seed="resume-shards",
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            first_consumer = QueueConsumer(20)
            first = StreamCacheProducer(output, cfg, first_consumer)
            first.add_training_document(_doc(1, [1, 2, 3]))
            first.drain_training()
            while not first_consumer.queue.empty():
                first_consumer.acknowledge(first_consumer.queue.get_nowait().block_id)
            state = first.checkpoint_state()
            previous = [item["filename"] for item in state["train_writer"]["shards"]]
            self.assertEqual(previous, ["train/train-000000.bin"])
            self.assertEqual(list(output.rglob("*.tmp")), [])

            resumed = StreamCacheProducer.from_state(output, cfg, state)
            resumed.add_training_document(_doc(2, [10, 11, 12]))
            manifest = resumed.finish()
            filenames = [item["filename"] for item in manifest["shards"]]
            self.assertIn("train/train-000000.bin", filenames)
            self.assertGreater(len(filenames), len(previous))

    def test_checkpoint_resume_matches_continuous_reference_at_same_boundary(self) -> None:
        cfg = StreamCacheConfig(
            context_length=3, sequences_per_block=2, target_shard_bytes=64,
            reader_workers=2, max_in_flight_work_items=3, per_cluster_queue_limit=10,
            prepared_block_queue_limit=20, prefetch_head_start=0,
            minimum_prefetched_source_tokens=0, minimum_populated_cluster_queues=1,
            weights=synthetic_test_weights(), scheduler_tie_break_seed="resume-equivalence",
        )
        before = [
            (False, _doc(1, [1, 2, 3])),
            (True, _doc(2, [4, 5])),
            (False, _doc(2, [6, 7, 8, 9])),
            (False, _doc(3, [10, 11])),
        ]
        after = [
            (True, _doc(4, [12, 13, 14])),
            (False, _doc(4, [15, 16, 17])),
            (False, _doc(1, [18, 19, 20, 21])),
            (True, _doc(5, [22, 23])),
        ]

        def feed(producer: StreamCacheProducer, documents) -> None:
            for validation, document in documents:
                if validation:
                    producer.add_validation_document(document)
                else:
                    producer.add_training_document(document)
                    producer.drain_training(force=False, maximum_documents=1)

        with tempfile.TemporaryDirectory() as reference_tmp, tempfile.TemporaryDirectory() as resumed_tmp:
            reference_root = Path(reference_tmp)
            reference = StreamCacheProducer(reference_root, cfg)
            feed(reference, before)
            reference.checkpoint_state()
            feed(reference, after)
            reference_manifest = reference.finish()

            resumed_root = Path(resumed_tmp)
            interrupted = StreamCacheProducer(resumed_root, cfg)
            feed(interrupted, before)
            state = interrupted.checkpoint_state()
            interrupted.close()
            resumed = StreamCacheProducer.from_state(resumed_root, cfg, state)
            feed(resumed, after)
            resumed_manifest = resumed.finish()

            self.assertEqual(resumed_manifest, reference_manifest)
            reference_files = sorted(path.relative_to(reference_root) for path in reference_root.rglob("*.bin"))
            resumed_files = sorted(path.relative_to(resumed_root) for path in resumed_root.rglob("*.bin"))
            self.assertEqual(resumed_files, reference_files)
            for relative in reference_files:
                self.assertEqual(
                    (resumed_root / relative).read_bytes(),
                    (reference_root / relative).read_bytes(),
                )

    def test_shard_writer_index_discovery_does_not_reuse_stale_tmp(self) -> None:
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
            stale_block = PackedSequence((1, 2, 3, 4), "a", "a", {1: 3})
            block = PreparedBlockBuilder(1)
            prepared = block.push(stale_block, split="train", cumulative_source_tokens=3)
            assert prepared is not None
            with self.assertRaises(RuntimeError):
                writer.write_block(prepared)

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

    def test_reader_batches_are_bounded_and_preserve_order(self) -> None:
        body = b"".join(
            (f'{{"cluster_id":1,"tokens":[{index},{index + 1}],"token_count":2}}\n').encode()
            for index in range(0, 100, 2)
        )
        plan = WorkPlan(schema_version=2, dataset="x", revision="r", source_glob="*", selection_seed="s",
                        region_bytes=len(body), source_files=(SourceFile("part", len(body)),),
                        work_items=(WorkItem(0, "part", 0, len(body)),), hash="0" * 64)
        class Reader:
            def file_size(self): return len(body)
            def read_range(self, offset, length): return body[offset:offset + length]
        batches = list(parallel_read_document_batches(
            plan, reader_factory=lambda _: Reader(), workers=1, max_in_flight=1,
            maximum_source_tokens_per_batch=6, maximum_documents_per_batch=3, maximum_bytes_per_batch=1000,
        ))
        self.assertGreater(len(batches), 1)
        self.assertTrue(all(batch.accepted_source_tokens <= 6 for batch in batches))
        self.assertEqual([doc.tokens[0] for batch in batches for _, doc in batch.records], list(range(0, 100, 2)))

    def test_reader_batches_round_robin_active_work_items(self) -> None:
        bodies = {
            "part_0": b"".join(
                b'{"cluster_id":1,"tokens":[1,2]}\n' for _ in range(5)
            ),
            "part_1": b"".join(
                b'{"cluster_id":2,"tokens":[3,4]}\n' for _ in range(5)
            ),
        }
        plan = WorkPlan(
            schema_version=2, dataset="x", revision="r", source_glob="*", selection_seed="s",
            region_bytes=max(map(len, bodies.values())),
            source_files=tuple(SourceFile(name, len(body)) for name, body in bodies.items()),
            work_items=tuple(
                WorkItem(index=index, filename=name, range_start=0, range_end=len(bodies[name]))
                for index, name in enumerate(bodies)
            ),
            hash="0" * 64,
        )

        class Reader:
            def __init__(self, body: bytes) -> None:
                self.body = body

            def file_size(self) -> int:
                return len(self.body)

            def read_range(self, offset: int, length: int) -> bytes:
                return self.body[offset:offset + length]

        batches = list(parallel_read_document_batches(
            plan,
            reader_factory=lambda source: Reader(bodies[source.path]),
            workers=2,
            max_in_flight=2,
            maximum_source_tokens_per_batch=4,
            maximum_documents_per_batch=2,
            maximum_bytes_per_batch=1000,
        ))
        self.assertEqual(
            [batch.work_item_index for batch in batches],
            [0, 1, 0, 1, 0, 1],
        )

    def test_parallel_documents_interleave_default_sized_dominated_batches(self) -> None:
        bodies = {
            "part_0": b"".join(
                b'{"cluster_id":1,"tokens":[1,2]}\n' for _ in range(4)
            ),
            "part_1": b"".join(
                b'{"cluster_id":1,"tokens":[3,4]}\n' for _ in range(4)
            ),
        }
        plan = WorkPlan(
            schema_version=2, dataset="x", revision="r", source_glob="*", selection_seed="s",
            region_bytes=max(map(len, bodies.values())),
            source_files=tuple(SourceFile(name, len(body)) for name, body in bodies.items()),
            work_items=tuple(
                WorkItem(index=index, filename=name, range_start=0, range_end=len(bodies[name]))
                for index, name in enumerate(bodies)
            ),
            hash="0" * 64,
        )

        class Reader:
            def __init__(self, body: bytes) -> None:
                self.body = body

            def file_size(self) -> int:
                return len(self.body)

            def read_range(self, offset: int, length: int) -> bytes:
                return self.body[offset:offset + length]

        def read(workers: int):
            return list(parallel_read_documents(
                plan,
                reader_factory=lambda source: Reader(bodies[source.path]),
                workers=workers,
                max_in_flight=2,
            ))

        one_worker = read(1)
        many_workers = read(4)
        self.assertEqual(
            [document.work_item_index for _, document in one_worker],
            [0, 1, 0, 1, 0, 1, 0, 1],
        )
        self.assertEqual(one_worker, many_workers)


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
