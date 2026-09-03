from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / "kaggle"
if str(KAGGLE) not in sys.path:
    sys.path.insert(0, str(KAGGLE))

import launch_sft  # noqa: E402
import sft_cli  # noqa: E402
import sft_scaled_runtime  # noqa: E402


class SFT100M10BWiringTests(unittest.TestCase):
    def test_profile_binds_completed_final_parent_without_selecting_recipe(self) -> None:
        profile = sft_cli.resolve_profile(100_000_000, 10_000_000_000)

        self.assertEqual(profile.model_label, "100M")
        self.assertEqual(profile.token_label, "10B")
        self.assertEqual(profile.parent_run_id, "100m-10b-deep-decay-from-step15500")
        self.assertEqual(profile.known_parent_consumed_tokens, 10_000_007_168)
        self.assertEqual(profile.parent_pointer, "latest")
        self.assertEqual(profile.parent_transport, "hf_storage_bucket")
        self.assertFalse(profile.recipe_ready)
        self.assertIsNone(profile.requested_sft_targets)
        self.assertEqual(profile.learning_rate, 0.0)
        self.assertIs(sft_cli.runtime_for(profile), sft_scaled_runtime)

    def test_dry_run_exposes_parent_wiring_and_pending_recipe(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(
                launch_sft.main(
                    ["train", "--model", "100M", "--tokens", "10B", "--dry-run"]
                ),
                0,
            )

        payload = json.loads(output.getvalue())
        self.assertEqual(payload["recipe_status"], "pending")
        self.assertIsNone(payload["sft_fraction"])
        self.assertIsNone(payload["requested_sft_targets"])
        self.assertIsNone(payload["learning_rate"])
        self.assertEqual(payload["known_exact_parent_consumed_tokens"], 10_000_007_168)
        self.assertEqual(payload["parent_pointer"], "latest")
        self.assertEqual(payload["parent_transport"], "hf_storage_bucket")
        self.assertEqual(payload["kaggle_training_topology"], "2xT4-DDP")

    def test_live_actions_fail_closed_until_recipe_is_accepted(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as caught:
            launch_sft.main(["train", "--model", "100M", "--tokens", "10B"])

        self.assertEqual(caught.exception.code, 2)
        message = stderr.getvalue()
        self.assertIn("ADR 0138", message)
        self.assertIn("scientific recipe pending", message)

    def test_profiles_lists_10b_as_pending(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(launch_sft.main(["profiles"]), 0)

        listing = output.getvalue()
        self.assertIn("parent_tokens=10B", listing)
        self.assertIn("100m-10b-deep-decay-from-step15500", listing)
        self.assertIn("fraction=pending", listing)
        self.assertIn("targets=pending", listing)


if __name__ == "__main__":
    unittest.main()
