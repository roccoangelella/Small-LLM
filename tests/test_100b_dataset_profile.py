"""Frozen public-HF profile and trainer horizon for the active 200M/100B run."""

from __future__ import annotations

import runpy
import unittest
from pathlib import Path

from dataset.incremental_frontier import build_run_contract
from dataset.qualification import get_profile, production_arguments

ROOT = Path(__file__).resolve().parents[1]


class Dataset100BProfileTests(unittest.TestCase):
    def test_100b_profile_reuses_the_10b_shard_strategy(self) -> None:
        ten = get_profile("modal-10b-b64")
        profile = get_profile("100b")

        self.assertEqual(profile.run_id, "100b-b64-dataset-001")
        self.assertEqual(profile.target_source_tokens, 100_000_000_000)
        self.assertEqual(profile.minimum_source_tokens, 90_000_000_000)
        self.assertEqual(profile.maximum_source_tokens, 110_000_000_000)
        self.assertEqual(profile.checkpoint_source_tokens, ten.checkpoint_source_tokens)
        self.assertEqual(profile.context_length, ten.context_length)
        self.assertEqual(profile.sequences_per_block, ten.sequences_per_block)
        self.assertEqual(profile.target_shard_bytes, ten.target_shard_bytes)
        self.assertTrue(profile.evict_remote_shards)
        self.assertTrue(profile.incremental_frontier)
        self.assertEqual(profile.nominal_training_tokens, 100_000_000_000)
        self.assertEqual(profile.training_validation_blocks, 16)
        self.assertFalse(profile.hf_bucket_private)
        self.assertFalse(profile.launch_concurrent_producer)

    def test_100b_build_is_locked_to_public_hf_incremental_upload(self) -> None:
        args = production_arguments(
            "100b",
            ["--weights-file", "dataset/climbmix_code_free_weights.json", "--output-dir", "out"],
        )

        self.assertEqual(args[args.index("--run-id") + 1], "100b-b64-dataset-001")
        self.assertEqual(args[args.index("--target-tokens") + 1], "100000000000")
        self.assertEqual(args[args.index("--nominal-training-tokens") + 1], "100000000000")
        self.assertEqual(args[args.index("--sequences-per-block") + 1], "64")
        self.assertEqual(args[args.index("--target-shard-bytes") + 1], str(1024**3))
        self.assertIn("--evict-remote-shards", args)
        self.assertIn("--incremental-frontier", args)
        self.assertIn("--public-hf-bucket", args)

    def test_100b_contract_has_exact_whole_block_horizon(self) -> None:
        profile = get_profile("100b")
        contract = build_run_contract(
            run_id=str(profile.run_id),
            nominal_training_tokens=int(profile.nominal_training_tokens or 0),
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

        self.assertEqual(contract["planned_train_blocks"], 762_940)
        self.assertEqual(contract["planned_train_target_tokens"], 100_000_071_680)
        # Dataset qualification retains its generic finite-horizon WSD report.
        # The active 200M scientific preset replaces this with ADR-0163 WSqD
        # inside trainer setup before checkpoint identity is computed.
        self.assertEqual(contract["trainer"]["steps"], 762_940)
        self.assertEqual(contract["trainer"]["full_block_target_tokens"], 131_072)
        self.assertEqual(contract["trainer"]["validation_blocks"], 16)
        self.assertEqual(contract["trainer"]["planned_target_tokens"], 100_000_071_680)

    def test_provider_launchers_stage_without_spawning_a_100b_producer(self) -> None:
        rolling = (ROOT / "providers" / "rolling_dataset.py").read_text(encoding="utf-8")
        self.assertIn("require_completed_frontier(store, run_id=profile.run_id)", rolling)
        self.assertIn("store.verify_bucket_visibility()", rolling)
        for provider in ("modal", "beam"):
            launch = (ROOT / provider / "launch.py").read_text(encoding="utf-8")
            with self.subTest(provider=provider):
                self.assertIn("dataset_profile.launch_concurrent_producer", launch)
                self.assertIn("100B runs require an explicit positive --max-steps-this-session budget", launch)
                expected_gpu = "DEFAULT_GPU" if provider == "modal" else '"RTX4090"'
                self.assertIn(f"gpu != {expected_gpu}", launch)

    def test_modal_and_beam_resolve_the_same_active_200m_100b_run(self) -> None:
        for provider in ("modal", "beam"):
            profiles = runpy.run_path(str(ROOT / provider / "profiles.py"))
            model, tokens = profiles["resolve_presets"]("200M", "100B")
            with self.subTest(provider=provider):
                self.assertEqual(model.trainer_size, "expanded")
                self.assertEqual(tokens.dataset_profile, "100b-b64")
                self.assertEqual(tokens.dataset_transport, "hf_rolling_shards")
                self.assertEqual(
                    profiles["canonical_run_id"](model, tokens),
                    "200m-100b-data-001",
                )
                self.assertEqual(
                    profiles["resolve_profile_selection"](
                        profile="200m-100b",
                        model=None,
                        tokens=None,
                    ),
                    ("200M", "100B"),
                )
                with self.assertRaises(ValueError):
                    profiles["resolve_presets"]("100M", "100B")
                with self.assertRaises(ValueError):
                    profiles["resolve_presets"]("200M", "10B")
                with self.assertRaisesRegex(
                    ValueError,
                    "either --profile or --model/--tokens",
                ):
                    profiles["resolve_profile_selection"](
                        profile="200M-100B",
                        model="200M",
                        tokens="100B",
                    )


if __name__ == "__main__":
    unittest.main()
