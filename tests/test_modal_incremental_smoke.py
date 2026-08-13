"""Contract tests for the opt-in live incremental Modal smoke."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

from dataset.incremental_frontier import build_run_contract

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modal"))
import incremental_smoke_support as smoke  # noqa: E402


class ModalIncrementalSmokeTests(unittest.TestCase):
    def test_geometry_and_standard_wsd(self) -> None:
        self.assertEqual(smoke.SMOKE_CONTEXT_LENGTH, 2048)
        self.assertEqual(smoke.SMOKE_SEQUENCES_PER_BLOCK, 64)
        self.assertEqual(smoke.SMOKE_BLOCK_BYTES, (2048 + 1) * 64 * 2)
        self.assertEqual(smoke.SMOKE_TARGET_SHARD_BYTES, smoke.SMOKE_BLOCK_BYTES * 16)
        self.assertEqual(smoke.SMOKE_FIRST_SEGMENT_STEPS, 16)
        self.assertGreater(smoke.SMOKE_TRAIN_BLOCKS, smoke.SMOKE_TOTAL_EXERCISED_STEPS)
        contract = build_run_contract(
            run_id="smoke-incremental-dataset-012345abcdef",
            nominal_training_tokens=smoke.SMOKE_NOMINAL_TRAINING_TOKENS,
            target_source_tokens=smoke.SMOKE_SOURCE_TARGET_TOKENS,
            minimum_source_tokens=smoke.SMOKE_SOURCE_MINIMUM_TOKENS,
            maximum_source_tokens=smoke.SMOKE_SOURCE_MAXIMUM_TOKENS,
            checkpoint_source_tokens=smoke.SMOKE_CHECKPOINT_SOURCE_TOKENS,
            context_length=smoke.SMOKE_CONTEXT_LENGTH,
            sequences_per_block=smoke.SMOKE_SEQUENCES_PER_BLOCK,
            target_shard_bytes=smoke.SMOKE_TARGET_SHARD_BYTES,
            configuration_hash="c" * 64,
            schema_hash="s" * 64,
            work_plan_hash="w" * 64,
            validation_blocks=smoke.SMOKE_VALIDATION_BLOCKS,
        )
        self.assertEqual(contract["planned_train_blocks"], 64)
        trainer = contract["trainer"]
        self.assertEqual(trainer["warmup_updates"], 16)
        self.assertEqual(trainer["stable_updates"], 35)
        self.assertEqual(trainer["decay_updates"], 13)

    def test_producer_and_trainer_arguments_are_isolated_live_smoke(self) -> None:
        args = smoke.producer_arguments(
            weights_file=Path("/repo/dataset/mixture_weights.json"),
            output_dir=Path("/cache/smoke"),
            dataset_run_id="smoke-incremental-dataset-012345abcdef",
            dataset_bucket_id="owner/datasets",
        )
        self.assertIn("--incremental-frontier", args)
        self.assertIn("--evict-remote-shards", args)
        self.assertNotIn("--allow-local-only", args)
        self.assertEqual(args[args.index("--training-validation-blocks") + 1], "1")
        self.assertEqual(
            int(args[args.index("--target-shard-bytes") + 1]), smoke.SMOKE_TARGET_SHARD_BYTES
        )

        command = smoke.wire_live_smoke_trainer_command(
            ["python", "-m", "trainer", "--remote-publish-every-steps", "0", "--wandb-mode", "disabled"],
            dataset_bucket_id="owner/datasets",
            dataset_run_id="smoke-incremental-dataset-012345abcdef",
            remote_manifest=Path("/run/transport.json"),
            checkpoint_repo_id="owner/model-incremental-smoke-012345abcdef",
        )
        self.assertGreater(smoke.SMOKE_REMOTE_PUBLISH_EVERY, smoke.SMOKE_TOTAL_EXERCISED_STEPS)
        self.assertEqual(
            int(command[command.index("--remote-publish-every-steps") + 1]),
            smoke.SMOKE_REMOTE_PUBLISH_EVERY,
        )
        self.assertEqual(command[command.index("--wandb-mode") + 1], "disabled")
        self.assertIn("--dataset-shard-bucket", command)
        self.assertIn("--remote-checkpoint-repo", command)
        self.assertIn("--remote-rolling-latest-only", command)
        self.assertNotIn("--remote-checkpoint-bucket", command)

    def test_identity_and_live_entrypoint_order(self) -> None:
        identity = smoke.smoke_identity("owner/small-llm", "012345abcdef")
        self.assertEqual(identity.dataset_run_id, "smoke-incremental-dataset-012345abcdef")
        self.assertEqual(identity.training_run_id, "smoke-incremental-train-012345abcdef")
        self.assertEqual(
            identity.checkpoint_repo_id,
            "owner/small-llm-incremental-smoke-012345abcdef",
        )
        with self.assertRaises(ValueError):
            smoke.validate_smoke_run_id("100m-10b-data-001")

        source = (ROOT / "modal" / "incremental_smoke.py").read_text(encoding="utf-8")
        main = source[source.index("@app.local_entrypoint()") :]
        producer = main.index("produce_remote.spawn")
        stage = main.index("stage_remote.spawn")
        supervise = main.index("await_stage_with_producer")
        first_gpu = main.index("train_segment_remote.spawn")
        move_local = main.index("move_local_checkpoint_aside_remote.remote")
        restage = main.index("stage_remote.remote")
        second_gpu = main.index("train_segment_remote.spawn", first_gpu + 1)
        self.assertLess(producer, stage)
        self.assertLess(stage, supervise)
        self.assertLess(supervise, first_gpu)
        self.assertLess(first_gpu, move_local)
        self.assertLess(move_local, restage)
        self.assertLess(restage, second_gpu)

    def test_gpu_and_split_guards_do_not_change_production_launcher(self) -> None:
        source = (ROOT / "modal" / "incremental_smoke.py").read_text(encoding="utf-8")
        train_pos = source.index("def train_segment_remote")
        train_decorator = source[source.rindex("@app.function(", 0, train_pos) : train_pos]
        self.assertIn('gpu="H100"', train_decorator)
        self.assertNotIn("retries=", train_decorator)
        for name in ("preflight_remote", "produce_remote", "stage_remote", "move_local_checkpoint_aside_remote"):
            position = source.index(f"def {name}")
            decorator = source[source.rindex("@app.function(", 0, position) : position]
            self.assertNotIn("gpu=", decorator)

        producer = source[source.index("def produce_remote") : source.index("def stage_remote")]
        self.assertLess(producer.index("forbidden ="), producer.index("VALIDATION_PROBABILITY ="))
        self.assertLess(
            producer.index("VALIDATION_PROBABILITY ="),
            producer.index("from dataset.production.cli import"),
        )
        production = (ROOT / "modal" / "launch.py").read_text(encoding="utf-8")
        self.assertNotIn("incremental_smoke", production)


if __name__ == "__main__":
    unittest.main()
