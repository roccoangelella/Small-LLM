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
from post_training.sft.s0_10pct_bundle import (  # noqa: E402
    EXPECTED_HELDOUT_MANIFEST_SHA256,
    FROZEN_HELDOUT_TARGETS,
    INSTRUCTION_TARGETS,
    PARENT_CONSUMED_TARGETS,
    PLANNED_TRAIN_SOURCE_TARGETS,
    REPLAY_TARGETS,
    TRAIN_INSTRUCTION_SOURCE_SHARES,
    TRAIN_TARGETS,
)


class SFT10PctCapacityAwareTests(unittest.TestCase):
    def _profile(self):
        base_profile = sft_cli.resolve_profile(100_000_000, 2_000_000_000)
        return sft_cli.with_sft_fraction(base_profile, Fraction(1, 10))

    def test_profile_identity_and_budget_are_isolated_from_four_percent(self) -> None:
        profile = self._profile()
        self.assertEqual(profile.sft_run_id, "100m-2b-sft-s0-10pct-peak3000-001")
        self.assertEqual(profile.wandb_run_id, "100m-2b-sft-s0-10pct-peak3000-001")
        self.assertEqual(profile.dataset_slug, "small-llm-100m-2b-sft-s0-10pct-001")
        self.assertEqual(profile.sft_fraction_numerator, 1)
        self.assertEqual(profile.sft_fraction_denominator, 10)
        self.assertEqual(profile.requested_sft_targets, TRAIN_TARGETS)
        self.assertEqual(TRAIN_TARGETS, 200_100_044)

    def test_capacity_aware_plan_preserves_top_level_85_15_contract(self) -> None:
        instruction_sources = {
            source: targets
            for source, targets in PLANNED_TRAIN_SOURCE_TARGETS.items()
            if source != "climbmix-replay"
        }
        self.assertEqual(sum(instruction_sources.values()), INSTRUCTION_TARGETS)
        self.assertEqual(PLANNED_TRAIN_SOURCE_TARGETS["climbmix-replay"], REPLAY_TARGETS)
        self.assertEqual(INSTRUCTION_TARGETS + REPLAY_TARGETS, TRAIN_TARGETS)
        self.assertAlmostEqual(sum(TRAIN_INSTRUCTION_SOURCE_SHARES.values()), 1.0, places=12)
        self.assertAlmostEqual(INSTRUCTION_TARGETS / TRAIN_TARGETS, 0.85, places=7)
        self.assertAlmostEqual(REPLAY_TARGETS / TRAIN_TARGETS, 0.15, places=7)

    def test_frozen_heldout_contract_is_pinned(self) -> None:
        self.assertEqual(FROZEN_HELDOUT_TARGETS, 2_106_316)
        self.assertEqual(
            EXPECTED_HELDOUT_MANIFEST_SHA256,
            {
                "validation": "26cb522729b4525498559d1ce131a181c30fd8fff573f3464e09030be803d09e",
                "test": "48e99ee51c201da398e227742ca7e023064a408c486cce16e20427d1ec7634d2",
            },
        )

    def test_runtime_routes_only_exact_10pct_parent_to_capacity_aware_builder(self) -> None:
        profile = self._profile()
        self.assertTrue(
            sft_scaled_runtime._is_capacity_aware_10pct(
                profile,
                parent_consumed_tokens=PARENT_CONSUMED_TARGETS,
            )
        )
        four_percent = sft_cli.resolve_profile(100_000_000, 2_000_000_000)
        self.assertFalse(
            sft_scaled_runtime._is_capacity_aware_10pct(
                four_percent,
                parent_consumed_tokens=PARENT_CONSUMED_TARGETS,
            )
        )

    def test_prepare_invokes_dedicated_builder_for_10pct(self) -> None:
        profile = self._profile()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            replay = root / "replay"
            replay.mkdir()
            (replay / "manifest.json").write_text("{}\n", encoding="utf-8")
            prepared = root / "prepared"
            prepared.mkdir()
            (prepared / "prepared-manifest.json").write_text("{}\n", encoding="utf-8")
            output = root / "bundle"
            worktree = root / "worktree"
            commands: list[list[str]] = []

            def capture(command: list[str], *, cwd: Path) -> int:
                commands.append(list(command))
                self.assertEqual(cwd, worktree)
                return 0

            with (
                mock.patch.object(sft_scaled_runtime.base, "_resolve_replay_root", return_value=replay),
                mock.patch.object(sft_scaled_runtime.base, "_prepare_worktree", return_value=worktree),
                mock.patch.object(sft_scaled_runtime.base, "_verify_existing_bundle_budget", return_value=False),
                mock.patch.object(sft_scaled_runtime.base, "_run", side_effect=capture),
            ):
                self.assertEqual(
                    sft_scaled_runtime.prepare(
                        profile,
                        replay_root=str(replay),
                        prepared_dir=str(prepared),
                        output_dir=str(output),
                        parent_consumed_tokens=PARENT_CONSUMED_TARGETS,
                        revision=None,
                    ),
                    0,
                )

            builder_commands = [
                command
                for command in commands
                if "post_training.sft.s0_10pct_bundle" in command
            ]
            self.assertEqual(len(builder_commands), 1)
            command = builder_commands[0]
            self.assertIn("--parent-consumed-tokens", command)
            parent_index = command.index("--parent-consumed-tokens")
            self.assertEqual(command[parent_index + 1], str(PARENT_CONSUMED_TARGETS))
            self.assertNotIn("post_training.sft.scaled_bundle", command)

    def test_existing_10pct_bundle_must_carry_recipe_and_frozen_identities(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = {
                "train_target_tokens_requested": TRAIN_TARGETS,
                "s0_scaling_recipe": {
                    "name": sft_scaled_runtime.TEN_PERCENT_RECIPE,
                    "expected_heldout_manifest_sha256": dict(EXPECTED_HELDOUT_MANIFEST_SHA256),
                },
                "splits": {
                    split: {"manifest_sha256": digest}
                    for split, digest in EXPECTED_HELDOUT_MANIFEST_SHA256.items()
                },
            }
            (root / "bundle-manifest.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            sft_scaled_runtime._verify_capacity_aware_bundle(root)

            payload["splits"]["validation"]["manifest_sha256"] = "0" * 64
            (root / "bundle-manifest.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                sft_scaled_runtime.base.RuntimeFailure,
                "frozen validation identity",
            ):
                sft_scaled_runtime._verify_capacity_aware_bundle(root)

    def test_train_uses_aggressive_10pct_worktree_and_wrapper(self) -> None:
        profile = self._profile()
        worktree = Path("/tmp/small-llm-sft-10pct-worktree")
        bundle = Path("/tmp/small-llm-sft-10pct-bundle")
        prepared_profiles = []
        captured: dict[str, object] = {}

        def prepare_worktree(selected_profile):
            prepared_profiles.append(selected_profile)
            return worktree

        def capture_run(command: list[str], *, cwd: Path) -> int:
            captured["command"] = list(command)
            captured["cwd"] = cwd
            return 0

        with (
            mock.patch.object(sft_scaled_runtime.base, "_prepare_worktree", side_effect=prepare_worktree),
            mock.patch.object(sft_scaled_runtime.base, "_find_bundle", return_value=bundle),
            mock.patch.object(sft_scaled_runtime, "_verify_published_10pct_training_bundle"),
            mock.patch.object(sft_scaled_runtime, "_require_stable_parent_artifact"),
            mock.patch.object(sft_scaled_runtime.base, "_wandb_preflight"),
            mock.patch.object(sft_scaled_runtime.base, "_run", side_effect=capture_run),
            mock.patch.dict("os.environ", {"HF_TOKEN": "test-token"}, clear=False),
        ):
            self.assertEqual(
                sft_scaled_runtime.train(
                    profile,
                    dataset_dir=str(bundle),
                    parent_repo_id="owner/parent",
                    checkpoint_repo_id="owner/sft",
                    max_steps_this_session=2,
                    wandb_entity=None,
                ),
                0,
            )

        self.assertEqual(len(prepared_profiles), 1)
        self.assertEqual(
            prepared_profiles[0].launch_commit,
            sft_scaled_runtime.TEN_PERCENT_TRAIN_COMMIT,
        )
        command = captured["command"]
        assert isinstance(command, list)
        self.assertTrue(
            any(str(item).endswith("kaggle/dual_t4_sft_10pct.py") for item in command)
        )
        self.assertIn("--learning-rate", command)
        lr_index = command.index("--learning-rate")
        self.assertEqual(float(command[lr_index + 1]), 3e-5)
        self.assertEqual(captured["cwd"], worktree)


if __name__ == "__main__":
    unittest.main()
