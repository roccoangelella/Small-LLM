from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / "kaggle"
if str(KAGGLE) not in sys.path:
    sys.path.insert(0, str(KAGGLE))

import sft_cli  # noqa: E402
import sft_scaled_runtime  # noqa: E402


class SFT10PctTrainingDatasetTests(unittest.TestCase):
    def _profile(self):
        base_profile = sft_cli.resolve_profile(100_000_000, 2_000_000_000)
        return sft_cli.with_sft_fraction(base_profile, Fraction(1, 10))

    def _write_bundle_manifest(self, root: Path) -> None:
        expected_heldout = {
            "validation": sft_scaled_runtime.TEN_PERCENT_PUBLISHED_SPLITS["validation"][
                "manifest_sha256"
            ],
            "test": sft_scaled_runtime.TEN_PERCENT_PUBLISHED_SPLITS["test"][
                "manifest_sha256"
            ],
        }
        payload = {
            "train_target_tokens_requested": sft_scaled_runtime.TEN_PERCENT_TRAIN_TARGETS,
            "s0_scaling_recipe": {
                "name": sft_scaled_runtime.TEN_PERCENT_RECIPE,
                "expected_heldout_manifest_sha256": expected_heldout,
            },
            "splits": {
                split: dict(fields)
                for split, fields in sft_scaled_runtime.TEN_PERCENT_PUBLISHED_SPLITS.items()
            },
        }
        (root / "bundle-manifest.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

    def test_published_dataset_identity_is_pinned(self) -> None:
        self.assertEqual(
            sft_scaled_runtime.TEN_PERCENT_PUBLISHED_TREE_SHA256,
            "c7550a377978231bfcc4d158ab11f8e2604e45921c5acb4e37e9557f12590b4d",
        )
        self.assertEqual(sft_scaled_runtime.TEN_PERCENT_PUBLISHED_FILE_COUNT, 22)
        self.assertEqual(sft_scaled_runtime.TEN_PERCENT_PUBLISHED_TOTAL_BYTES, 773_987_135)
        self.assertEqual(
            sft_scaled_runtime.TEN_PERCENT_PUBLISHED_SPLITS["train"],
            {
                "loss_bearing_target_tokens": 200_099_738,
                "manifest_sha256": "feefc3244bd8a2f369eec85e4a95410c2daf479016c04cf02c8042ca5a4010d3",
                "build_report_sha256": "8a131988c43349fb360f56dd41f7f552e9c1533c2550701db67b37ece6e820d7",
            },
        )
        self.assertEqual(
            sft_scaled_runtime.TEN_PERCENT_PUBLISHED_SPLITS["validation"],
            {
                "loss_bearing_target_tokens": 2_105_945,
                "manifest_sha256": "26cb522729b4525498559d1ce131a181c30fd8fff573f3464e09030be803d09e",
                "build_report_sha256": "37e3c4d98d1e7ed1ec077e0e92b7d79327c4ee2b473f0c4e86f0ec5e4d6c324d",
            },
        )

    def test_published_10pct_bundle_accepts_exact_training_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_bundle_manifest(root)
            sft_scaled_runtime._verify_published_10pct_training_bundle(root)

    def test_published_10pct_bundle_rejects_wrong_train_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_bundle_manifest(root)
            path = root / "bundle-manifest.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["splits"]["train"]["manifest_sha256"] = "0" * 64
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                sft_scaled_runtime.base.RuntimeFailure,
                "training dataset identity mismatch",
            ):
                sft_scaled_runtime._verify_published_10pct_training_bundle(root)

    def test_10pct_train_preflights_verified_dataset_and_uses_isolated_run_id(self) -> None:
        profile = self._profile()
        worktree = Path("/tmp/small-llm-sft-worktree")
        bundle = Path("/tmp/small-llm-100m-2b-sft-s0-10pct-001")
        captured: dict[str, object] = {}

        def capture_run(command: list[str], *, cwd: Path) -> int:
            captured["command"] = list(command)
            captured["cwd"] = cwd
            return 0

        with (
            mock.patch.object(sft_scaled_runtime.base, "_prepare_worktree", return_value=worktree),
            mock.patch.object(sft_scaled_runtime.base, "_find_bundle", return_value=bundle) as find_bundle,
            mock.patch.object(
                sft_scaled_runtime,
                "_verify_published_10pct_training_bundle",
            ) as dataset_preflight,
            mock.patch.object(sft_scaled_runtime, "_require_stable_parent_artifact"),
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
                    max_steps_this_session=None,
                    wandb_entity=None,
                ),
                0,
            )

        find_bundle.assert_called_once_with(None, profile)
        dataset_preflight.assert_called_once_with(bundle)
        command = captured["command"]
        self.assertIsInstance(command, list)
        assert isinstance(command, list)
        run_id_index = command.index("--sft-run-id")
        self.assertEqual(
            command[run_id_index + 1],
            "100m-2b-sft-s0-10pct-longpeak-001",
        )
        fraction_numerator = command.index("--sft-fraction-numerator")
        fraction_denominator = command.index("--sft-fraction-denominator")
        self.assertEqual(command[fraction_numerator + 1], "1")
        self.assertEqual(command[fraction_denominator + 1], "10")
        dataset_index = command.index("--dataset-dir")
        self.assertEqual(command[dataset_index + 1], str(bundle))


if __name__ == "__main__":
    unittest.main()
