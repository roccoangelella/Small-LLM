"""Frozen geometry and producer transport tests for the 100M/10B dataset."""

from __future__ import annotations

import unittest

from dataset.incremental_frontier import build_run_contract
from dataset.qualification import get_profile, production_arguments


class Dataset10BProfileTests(unittest.TestCase):
    def test_modal_10b_profile_freezes_block64_one_gib_hf_shards(self) -> None:
        profile = get_profile("modal-10b-b64")

        self.assertEqual(profile.run_id, "modal-10b-b64-dataset-001")
        self.assertEqual(profile.target_source_tokens, 10_000_000_000)
        self.assertEqual(profile.minimum_source_tokens, 9_000_000_000)
        self.assertEqual(profile.maximum_source_tokens, 11_000_000_000)
        self.assertEqual(profile.checkpoint_source_tokens, 500_000_000)
        self.assertEqual(profile.context_length, 2048)
        self.assertEqual(profile.sequences_per_block, 64)
        self.assertEqual(profile.target_shard_bytes, 1024**3)
        self.assertTrue(profile.evict_remote_shards)
        self.assertTrue(profile.incremental_frontier)
        self.assertEqual(profile.nominal_training_tokens, 10_000_000_000)
        self.assertEqual(profile.training_validation_blocks, 16)

        block_bytes = (2048 + 1) * 64 * 2
        self.assertEqual((1024**3) // block_bytes, 4094)
        self.assertEqual(4094 * block_bytes, 1_073_741_568)
        self.assertEqual(1024**3 - 4094 * block_bytes, 256)

    def test_modal_10b_profile_forces_incremental_remote_eviction(self) -> None:
        args = production_arguments("modal-10b-b64", ["--weights-file", "weights.json"])

        self.assertEqual(args[args.index("--run-id") + 1], "modal-10b-b64-dataset-001")
        self.assertEqual(args[args.index("--target-shard-bytes") + 1], str(1024**3))
        self.assertEqual(args[args.index("--sequences-per-block") + 1], "64")
        self.assertEqual(args[args.index("--nominal-training-tokens") + 1], "10000000000")
        self.assertEqual(args[args.index("--training-validation-blocks") + 1], "16")
        self.assertNotIn("--remote-backend", args)
        self.assertIn("--evict-remote-shards", args)
        self.assertIn("--incremental-frontier", args)

    def test_modal_10b_contract_has_exact_prelaunch_horizon(self) -> None:
        profile = get_profile("modal-10b-b64")
        self.assertIsNotNone(profile.run_id)
        self.assertIsNotNone(profile.nominal_training_tokens)
        contract = build_run_contract(
            run_id=str(profile.run_id),
            nominal_training_tokens=int(profile.nominal_training_tokens),
            target_source_tokens=profile.target_source_tokens,
            minimum_source_tokens=profile.minimum_source_tokens,
            maximum_source_tokens=profile.maximum_source_tokens,
            checkpoint_source_tokens=profile.checkpoint_source_tokens,
            context_length=profile.context_length,
            sequences_per_block=profile.sequences_per_block,
            target_shard_bytes=profile.target_shard_bytes,
            configuration_hash="a" * 64,
            schema_hash="b" * 64,
            work_plan_hash="c" * 64,
            validation_blocks=profile.training_validation_blocks,
        )
        self.assertEqual(contract["planned_train_blocks"], 76_294)
        self.assertEqual(contract["planned_train_target_tokens"], 10_000_007_168)


if __name__ == "__main__":
    unittest.main()
