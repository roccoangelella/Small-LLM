"""Modal contracts for Bucket latest plus dedicated recreate-only model best."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MODAL = ROOT / "modal"
if str(MODAL) not in sys.path:
    sys.path.insert(0, str(MODAL))

import model_repo_checkpoint as transport  # noqa: E402
from trainer.cli_args import parse_args  # noqa: E402


class ModalSplitCheckpointTransportTests(unittest.TestCase):
    def test_online_modal_command_keeps_bucket_latest_and_adds_dedicated_best(self) -> None:
        base_command = [
            "python",
            "-m",
            "trainer",
            "--remote-checkpoint-bucket",
            "owner/base-checkpoints",
            "--remote-create-bucket",
            "--remote-rolling-latest-only",
        ]
        with (
            patch.object(transport, "_ORIGINAL_TRAINER_COMMAND", return_value=base_command.copy()),
            patch.dict(os.environ, {"SMALL_LLM_HF_REPO_ID": "owner/base"}, clear=False),
        ):
            command = transport._trainer_command_split_store(
                online=True,
                wandb_run_id="run-001",
            )

        self.assertIn("--remote-checkpoint-bucket", command)
        self.assertNotIn("--remote-checkpoint-repo", command)
        self.assertEqual(
            command[command.index("--best-model-repo") + 1],
            "owner/base-best-run-001",
        )
        self.assertIn("--best-model-recreate", command)

    def test_best_repo_override_must_remain_dedicated_to_run(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SMALL_LLM_HF_REPO_ID": "owner/base",
                "SMALL_LLM_HF_BEST_MODEL_REPO_ID": "owner/shared",
            },
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "not dedicated"):
                transport._dedicated_best_repo_id("run-001")

    def test_trainer_parser_requires_recreate_for_best_model_repo(self) -> None:
        base = [
            "--dataset-dir",
            "/tmp/data",
            "--checkpoint-dir",
            "/tmp/checkpoints",
            "--steps",
            "1",
            "--validation-blocks",
            "1",
            "--best-model-repo",
            "owner/base-best-run-001",
        ]
        with self.assertRaisesRegex(SystemExit, "requires --best-model-recreate"):
            parse_args(base)
        parsed = parse_args(base + ["--best-model-recreate"])
        self.assertEqual(parsed.best_model_repo, "owner/base-best-run-001")
        self.assertTrue(parsed.best_model_recreate)


if __name__ == "__main__":
    unittest.main()
