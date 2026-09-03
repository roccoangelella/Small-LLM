from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
KAGGLE_SRC = ROOT / "kaggle" / "src"
if str(KAGGLE_SRC) not in sys.path:
    sys.path.insert(0, str(KAGGLE_SRC))

import dual_t4_sft  # noqa: E402
import launch_sft  # noqa: E402
import sft_cli  # noqa: E402
import sft_scaled_runtime  # noqa: E402
from post_training.sft.bundle import sft_budget_from_parent  # noqa: E402
from post_training.sft.config import DEFAULT_INSTRUCTION_SOURCE_SHARES, SFTDataConfig  # noqa: E402


class SFT100M2BKaggleTests(unittest.TestCase):
    def test_profile_uses_completed_parent_and_four_percent_budget(self) -> None:
        profile = sft_cli.resolve_profile(100_000_000, 2_000_000_000)
        self.assertEqual(profile.parent_run_id, "100m-2b-data-001")
        self.assertEqual(profile.sft_run_id, "100m-2b-sft-s0-001")
        self.assertEqual(profile.known_parent_consumed_tokens, 2_001_000_448)
        self.assertEqual(profile.sft_fraction_numerator, 4)
        self.assertEqual(profile.sft_fraction_denominator, 100)
        self.assertEqual(profile.microbatch_size, 2)
        self.assertEqual(
            profile.launch_commit,
            "ca16b22905ebedc5925ab0abb9c40125254f1e1c",
        )
        self.assertEqual(profile.requested_sft_targets, 80_040_017)
        self.assertEqual(
            sft_budget_from_parent(2_001_000_448, numerator=4, denominator=100),
            80_040_017,
        )

    def test_stratification_is_unchanged(self) -> None:
        config = SFTDataConfig()
        self.assertEqual(config.instruction_share, 0.85)
        self.assertEqual(config.replay_share, 0.15)
        self.assertEqual(
            DEFAULT_INSTRUCTION_SOURCE_SHARES,
            {
                "smol-magpie-ultra-short": 0.75,
                "smol-contraints": 0.10,
                "smollm-rewrite-30k": 0.075,
                "smol-summarize-20k": 0.075,
            },
        )

    def test_canonical_dry_run_declares_dual_t4_topology(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(
                launch_sft.main(
                    ["train", "--model", "100M", "--tokens", "2B", "--dry-run"]
                ),
                0,
            )
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["sft_fraction"], 0.04)
        self.assertEqual(payload["requested_sft_targets"], 80_040_017)
        self.assertEqual(payload["kaggle_training_topology"], "2xT4-DDP")
        self.assertEqual(payload["microbatch_size"], 2)

    def test_train_pins_the_qualified_dual_t4_runtime(self) -> None:
        profile = sft_cli.resolve_profile(100_000_000, 2_000_000_000)
        worktree = Path("/tmp/small-llm-sft-worktree")
        runner = worktree / "kaggle" / "src" / "dual_t4_sft.py"
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
                return_value=Path("/tmp/small-llm-sft-bundle"),
            ),
            mock.patch.object(sft_scaled_runtime, "_runner_path", return_value=runner),
            mock.patch.object(sft_scaled_runtime, "_require_stable_parent_artifact") as parent_preflight,
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
        self.assertEqual(command[0], "env")
        for setting in sft_scaled_runtime.KAGGLE_SFT_PROCESS_ENV:
            self.assertIn(setting, command)
        self.assertIn("SMALL_LLM_DISABLE_OPTIMIZER_TELEMETRY=1", command)
        self.assertIn("MALLOC_ARENA_MAX=2", command)
        python_index = command.index("python")
        self.assertLess(command.index("torch==2.10.0"), python_index)
        self.assertLess(command.index("triton==3.6.0"), python_index)
        self.assertLess(command.index("fla-core==0.5.2"), python_index)
        self.assertLess(
            command.index("https://download.pytorch.org/whl/cu128"),
            python_index,
        )
        self.assertIn(str(runner), command)
        microbatch_index = command.index("--microbatch-size")
        self.assertEqual(command[microbatch_index + 1], "2")
        validation_index = command.index("--validation-blocks")
        behavior_index = command.index("--behavior-cases")
        self.assertIn("--remote-rolling-latest-only", command)
        self.assertEqual(
            command[validation_index + 1],
            str(sft_scaled_runtime.INLINE_VALIDATION_BLOCKS),
        )
        self.assertEqual(
            command[behavior_index + 1],
            str(sft_scaled_runtime.INLINE_BEHAVIOR_CASES),
        )
        self.assertEqual(sft_scaled_runtime.INLINE_VALIDATION_BLOCKS, 1)
        self.assertEqual(sft_scaled_runtime.INLINE_BEHAVIOR_CASES, 2)
        self.assertEqual(captured["cwd"], worktree)
        parent_preflight.assert_called_once_with(
            repo_id="owner/parent",
            run_id="100m-2b-data-001",
            token="test-token",
        )

    def test_parent_preflight_rejects_a_repository_for_another_model(self) -> None:
        api = mock.Mock()
        api.list_repo_files.return_value = [
            "models/20m-500m-data-001/artifact.json",
            "models/20m-500m-data-001/step-00015264/checkpoint.json",
        ]
        with self.assertRaisesRegex(
            sft_scaled_runtime.base.RuntimeFailure,
            "contains no stable artifact for run '100m-2b-data-001'",
        ):
            sft_scaled_runtime._require_stable_parent_artifact(
                repo_id="owner/20m-models",
                run_id="100m-2b-data-001",
                token="test-token",
                api=api,
            )

    def test_parent_preflight_accepts_the_stable_pointer(self) -> None:
        api = mock.Mock()
        api.list_repo_files.return_value = [
            "models/100m-2b-data-001/artifact.json",
        ]
        output = io.StringIO()
        with redirect_stdout(output):
            sft_scaled_runtime._require_stable_parent_artifact(
                repo_id="owner/100m-models",
                run_id="100m-2b-data-001",
                token="test-token",
                api=api,
            )
        self.assertIn("status=available", output.getvalue())

    def test_variable_sft_rows_partition_without_duplication(self) -> None:
        self.assertEqual(dual_t4_sft.SUPPORTED_MICROBATCH_SIZES, (1, 2, 4))
        for count in range(1, 18):
            left = dual_t4_sft._rank_row_indices(count, 0, 2)
            right = dual_t4_sft._rank_row_indices(count, 1, 2)
            self.assertEqual(sorted((*left, *right)), list(range(count)))
            self.assertTrue(set(left).isdisjoint(right))
            self.assertLessEqual(abs(len(left) - len(right)), 1)

    def test_long_rank_zero_side_effects_use_a_bounded_cpu_control_group(self) -> None:
        distributed = mock.Mock()
        group = object()
        distributed.new_group.return_value = group

        self.assertIs(dual_t4_sft._new_control_group(distributed), group)
        kwargs = distributed.new_group.call_args.kwargs
        self.assertEqual(kwargs["backend"], "gloo")
        self.assertEqual(
            kwargs["timeout"].total_seconds(),
            dual_t4_sft.CONTROL_GROUP_TIMEOUT_SECONDS,
        )

        dual_t4_sft._control_barrier(distributed, group)
        distributed.barrier.assert_called_once_with(group=group)

    def test_secondary_rank_disables_upload_and_rolling_cleanup(self) -> None:
        original_publish = mock.Mock()
        original_cleanup = mock.Mock()
        module = SimpleNamespace(
            CheckpointCoordinator=SimpleNamespace(publish=original_publish),
            cleanup_remote_publication=original_cleanup,
        )

        dual_t4_sft._disable_secondary_remote_side_effects(module)

        self.assertIsNone(module.CheckpointCoordinator.publish(object()))
        self.assertIsNone(module.cleanup_remote_publication(object(), checkpoint_id="step-00000250"))
        original_publish.assert_not_called()
        original_cleanup.assert_not_called()


if __name__ == "__main__":
    unittest.main()
