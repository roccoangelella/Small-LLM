from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


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

    def test_canonical_run_ids_are_arm_specific(self) -> None:
        self.assertEqual(
            rsft_runtime.default_run_id("atomic"),
            "100m-2b-rsft-r0-atomic-pilot-001",
        )
        self.assertEqual(
            rsft_runtime.default_run_id("textual"),
            "100m-2b-rsft-r0-textual-pilot-001",
        )

    def test_auto_preparation_plan_uses_committed_corpus_and_small_pilot_blocks(self) -> None:
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

    def test_minimal_atomic_cli_dry_run_requires_no_manual_dataset_arguments(self) -> None:
        result = rsft_cli.main(
            [
                "train",
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
