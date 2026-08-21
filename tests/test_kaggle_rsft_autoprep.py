from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
KAGGLE = REPO / "kaggle"
if str(KAGGLE) not in sys.path:
    sys.path.insert(0, str(KAGGLE))

import rsft_cli
import rsft_prepare
import rsft_runtime


class KaggleRSFTAutoPreparationTests(unittest.TestCase):
    def test_frozen_reasoning_token_spec_uses_qwen_style_thinking_markers(self) -> None:
        payload = json.loads(
            (REPO / "post_training" / "R-SFT" / "reasoning-tokens.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            payload,
            {
                "reasoning_start": "<think>",
                "reasoning_end": "</think>",
                "answer_start": "<answer>",
            },
        )

    def test_historical_pilot_run_ids_are_arm_specific(self) -> None:
        self.assertEqual(
            rsft_runtime.default_pilot_run_id("atomic"),
            "100m-2b-rsft-r0-atomic-pilot-001",
        )
        self.assertEqual(
            rsft_runtime.default_pilot_run_id("textual"),
            "100m-2b-rsft-r0-textual-pilot-001",
        )
        self.assertEqual(rsft_runtime.PRODUCTION_RUN_ID, "100m-2b-rsft-r0-16716-001")
        self.assertEqual(rsft_runtime.ACCEPTED_RUN_ID, "100m-2b-rsft-r0-12306-001")

    def test_auto_preparation_plan_is_pilot_only_and_uses_small_blocks(self) -> None:
        plan = rsft_prepare.preparation_plan(worktree=REPO)
        self.assertEqual(
            Path(str(plan["reasoning_jsonl"])),
            (REPO / "artifacts" / "rsft-r0-pilot-630" / "generation" / "reasoning.jsonl").resolve(),
        )
        self.assertEqual(
            Path(str(plan["reasoning_token_spec"])),
            (REPO / "post_training" / "R-SFT" / "reasoning-tokens.json").resolve(),
        )
        self.assertEqual(plan["optimizer_target_tokens"], 2_048)
        self.assertEqual(plan["passes"], 1)
        self.assertEqual(
            plan["s0_kaggle_handle"],
            "roccoangelella/small-llm-100m-2b-sft-s0-001",
        )

    def test_private_s0_download_bypasses_noninteractive_kaggle_attach_resolver(self) -> None:
        fake = mock.Mock(returncode=1, stdout="simulated failure\n")
        with mock.patch.object(rsft_prepare.subprocess, "run", fake):
            with self.assertRaises(rsft_prepare.base.RuntimeFailure):
                rsft_prepare._download_s0_bundle(
                    worktree=REPO,
                    handle="owner/private-s0",
                )
        self.assertEqual(fake.call_count, 1)
        kwargs = fake.call_args.kwargs
        self.assertEqual(kwargs["env"]["DISABLE_KAGGLE_CACHE"], "true")
        self.assertEqual(kwargs["env"]["PYTHONUNBUFFERED"], "1")
        self.assertEqual(kwargs["env"]["UV_LINK_MODE"], "copy")

    def test_production_reasoning_manifest_is_sha_pinned(self) -> None:
        reasoning = (REPO / rsft_prepare.PRODUCTION_REASONING_RELATIVE_PATH).resolve()
        loaded = rsft_prepare._require_production_reasoning_manifest(reasoning)
        self.assertEqual(loaded["output_sha256"], rsft_prepare.PRODUCTION_REASONING_SHA256)
        self.assertEqual(loaded["combined_rows"], 16_716)
        self.assertEqual(loaded["adapted_superior_rows"], 8_403)
        self.assertEqual(loaded["accepted_keep_rewrites"], 8_473)
        self.assertEqual(loaded["duplicate_rewrite_exclusions"], 70)


    def test_production_preparation_plan_uses_committed_superior_instruction_corpus(self) -> None:
        plan = rsft_prepare.production_preparation_plan(worktree=REPO)
        self.assertEqual(
            Path(str(plan["reasoning_jsonl"])),
            (REPO / "artifacts" / "rsft-superior-instruction-r0-expanded" / "reasoning.jsonl").resolve(),
        )
        self.assertEqual(plan["optimizer_target_tokens"], 32_768)
        self.assertEqual(plan["heldout_fraction_per_split"], 0.01)
        self.assertEqual(plan["reasoning_share"], 0.90)
        self.assertEqual(plan["s0_retention_share"], 0.10)
        self.assertEqual(plan["passes"], 1)

    def test_minimal_production_dry_run_requires_no_manual_dataset_arguments(self) -> None:
        result = rsft_cli.main(
            [
                "train",
                "--model",
                "100M",
                "--tokens",
                "2B",
                "--dry-run",
            ]
        )
        self.assertEqual(result, 0)

    def test_minimal_atomic_ablation_dry_run_requires_no_manual_dataset_arguments(self) -> None:
        result = rsft_cli.main(
            [
                "ablation",
                "--model",
                "100M",
                "--tokens",
                "2B",
                "--delimiter-format",
                "atomic",
                "--dry-run",
            ]
        )
        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
