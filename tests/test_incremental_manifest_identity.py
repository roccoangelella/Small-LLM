"""Regression tests for mutable producer state staying out of trainer identity."""

from __future__ import annotations

import copy
import unittest

from dataset.incremental_stage import _stable_consumer_manifest


class IncrementalManifestIdentityTests(unittest.TestCase):
    def test_producer_completion_does_not_change_trainer_manifest(self) -> None:
        contract = {
            "version": 1,
            "run_id": "dataset-001",
            "contract_sha256": "c" * 64,
            "schema_version": 2,
            "sequence_format": "context_plus_one",
            "context_length": 3,
            "stored_tokens_per_sequence": 4,
            "sequences_per_block": 2,
            "target_shard_bytes": 1024,
            "work_plan_hash": "w" * 64,
            "configuration_hash": "a" * 64,
            "schema_hash": "b" * 64,
            "planned_train_blocks": 4,
            "source_policy": {
                "target_source_tokens": 100,
                "minimum_source_tokens": 90,
                "maximum_source_tokens": 110,
                "checkpoint_source_tokens": 20,
            },
            "trainer": {
                "steps": 4,
                "validation_blocks": 1,
            },
            "frontier_policy": {
                "minimum_ready_train_shards_before_gpu": 2,
            },
        }
        train = [
            {
                "filename": "train/train-000000.bin",
                "split": "train",
                "byte_size": 16,
                "checksum": "0" * 64,
                "first_block_id": 0,
                "last_block_id": 0,
                "sequence_count": 2,
            },
            {
                "filename": "train/train-000001.bin",
                "split": "train",
                "byte_size": 16,
                "checksum": "1" * 64,
                "first_block_id": 1,
                "last_block_id": 1,
                "sequence_count": 2,
            },
        ]
        validation = [
            {
                "filename": "validation/validation-000000.bin",
                "split": "validation",
                "byte_size": 16,
                "checksum": "2" * 64,
                "first_block_id": 0,
                "last_block_id": 0,
                "sequence_count": 2,
            }
        ]
        active = {
            "ready_train_shards": train,
            "frozen_validation_shards": validation,
            "producer_complete": False,
        }
        complete = copy.deepcopy(active)
        complete["producer_complete"] = True
        complete["final_manifest_sha256"] = "f" * 64

        self.assertEqual(
            _stable_consumer_manifest(contract, active),
            _stable_consumer_manifest(contract, complete),
        )


if __name__ == "__main__":
    unittest.main()
