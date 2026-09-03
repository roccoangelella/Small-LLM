from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
KAGGLE_SRC = ROOT / "kaggle" / "src"
if str(KAGGLE_SRC) not in sys.path:
    sys.path.insert(0, str(KAGGLE_SRC))

import dual_t4_sft_10b_same_data  # noqa: E402
import launch_sft  # noqa: E402
import sft_100m_10b  # noqa: E402
import sft_cli  # noqa: E402
import sft_scaled_runtime  # noqa: E402


class SFT100M10BWiringTests(unittest.TestCase):
    def test_profile_reuses_2b_10pct_sft_dataset_as_absolute_budget(self) -> None:
        profile = sft_cli.resolve_profile(100_000_000, 10_000_000_000)

        self.assertEqual(profile.model_label, "100M")
        self.assertEqual(profile.token_label, "10B")
        self.assertEqual(profile.parent_run_id, "100m-10b-deep-decay-from-step15500")
        self.assertEqual(profile.known_parent_consumed_tokens, 10_000_007_168)
        self.assertEqual(profile.parent_pointer, "latest")
        self.assertEqual(profile.parent_transport, "hf_storage_bucket")
        self.assertTrue(profile.recipe_ready)
        self.assertFalse(profile.allow_sft_fraction_override)
        self.assertEqual(profile.dataset_slug, "small-llm-100m-2b-sft-s0-10pct-001")
        self.assertEqual(profile.requested_sft_targets, 200_100_044)
        self.assertEqual(profile.sft_fraction_numerator, 200_100_044)
        self.assertEqual(profile.sft_fraction_denominator, 10_000_007_168)
        self.assertEqual(profile.learning_rate, 3e-5)
        self.assertEqual(profile.launch_commit, sft_100m_10b.IMPLEMENTATION_COMMIT)
        self.assertIs(sft_cli.runtime_for(profile), sft_scaled_runtime)

    def test_dry_run_exposes_equal_sft_token_comparison(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(
                launch_sft.main(
                    ["train", "--model", "100M", "--tokens", "10B", "--dry-run"]
                ),
                0,
            )

        payload = json.loads(output.getvalue())
        self.assertEqual(payload["recipe_status"], "ready")
        self.assertAlmostEqual(payload["sft_fraction"], 200_100_044 / 10_000_007_168)
        self.assertEqual(payload["requested_sft_targets"], 200_100_044)
        self.assertEqual(payload["learning_rate"], 3e-5)
        self.assertEqual(payload["dataset_slug"], "small-llm-100m-2b-sft-s0-10pct-001")
        self.assertEqual(payload["known_exact_parent_consumed_tokens"], 10_000_007_168)
        self.assertEqual(payload["parent_pointer"], "latest")
        self.assertEqual(payload["parent_transport"], "hf_storage_bucket")
        self.assertEqual(payload["kaggle_training_topology"], "2xT4-DDP")

    def test_fraction_override_is_rejected_for_same_data_experiment(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as caught:
            launch_sft.main(
                [
                    "train",
                    "--model",
                    "100M",
                    "--tokens",
                    "10B",
                    "--sft-fraction",
                    "4%",
                    "--dry-run",
                ]
            )

        self.assertEqual(caught.exception.code, 2)
        message = stderr.getvalue()
        self.assertIn("fixed absolute corpus budget", message)
        self.assertIn("do not pass --sft-fraction", message)

    def test_profiles_lists_10b_as_ready_absolute_budget(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(launch_sft.main(["profiles"]), 0)

        listing = output.getvalue()
        self.assertIn("parent_tokens=10B", listing)
        self.assertIn("100m-10b-deep-decay-from-step15500", listing)
        self.assertIn("100m-10b-sft-s0-2b10pct-data-001", listing)
        self.assertIn("fraction=2.00%", listing)
        self.assertIn("targets=200100044", listing)

    def test_train_routes_to_10b_same_data_runner_and_bucket_parent(self) -> None:
        profile = sft_cli.resolve_profile(100_000_000, 10_000_000_000)
        worktree = Path("/tmp/small-llm-sft-worktree")
        runner = worktree / "kaggle" / "src" / "dual_t4_sft_10b_same_data.py"
        captured: dict[str, object] = {}

        def capture_run(command: list[str], *, cwd: Path) -> int:
            captured["command"] = command
            captured["cwd"] = cwd
            return 0

        with (
            mock.patch.object(sft_scaled_runtime.base, "_prepare_worktree", return_value=worktree),
            mock.patch.object(
                sft_scaled_runtime.base,
                "_find_bundle",
                return_value=Path("/tmp/small-llm-100m-2b-sft-s0-10pct-001-bundle"),
            ),
            mock.patch.object(sft_scaled_runtime, "_runner_path", return_value=runner),
            mock.patch.object(sft_scaled_runtime, "_verify_published_10pct_training_bundle"),
            mock.patch.object(sft_scaled_runtime, "_require_parent_artifact") as parent_preflight,
            mock.patch.object(sft_scaled_runtime.base, "_wandb_preflight"),
            mock.patch.object(sft_scaled_runtime.base, "_run", side_effect=capture_run),
            mock.patch.dict("os.environ", {"HF_TOKEN": "test-token"}, clear=False),
        ):
            self.assertEqual(
                sft_scaled_runtime.train(
                    profile,
                    dataset_dir=None,
                    parent_repo_id="owner/parent",
                    checkpoint_repo_id="owner/sft",
                    max_steps_this_session=20,
                    wandb_entity=None,
                ),
                0,
            )

        command = captured["command"]
        self.assertIsInstance(command, list)
        assert isinstance(command, list)
        self.assertIn(str(runner), command)
        pointer_index = command.index("--parent-pointer")
        self.assertEqual(command[pointer_index + 1], "latest")
        numerator_index = command.index("--sft-fraction-numerator")
        denominator_index = command.index("--sft-fraction-denominator")
        self.assertEqual(command[numerator_index + 1], str(dual_t4_sft_10b_same_data.EXPECTED_SFT_TARGETS))
        self.assertEqual(command[denominator_index + 1], str(dual_t4_sft_10b_same_data.EXPECTED_PARENT_TARGETS))
        run_index = command.index("--sft-run-id")
        self.assertEqual(command[run_index + 1], "100m-10b-sft-s0-2b10pct-data-001")
        self.assertEqual(captured["cwd"], worktree)
        parent_preflight.assert_called_once_with(
            profile,
            repo_id="owner/parent",
            run_id="100m-10b-deep-decay-from-step15500",
            token="test-token",
        )


if __name__ == "__main__":
    unittest.main()
