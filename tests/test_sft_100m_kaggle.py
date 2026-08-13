from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / "kaggle"
if str(KAGGLE) not in sys.path:
    sys.path.insert(0, str(KAGGLE))

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
        self.assertEqual(payload["microbatch_size"], 4)

    def test_train_pins_the_qualified_dual_t4_runtime(self) -> None:
        profile = sft_cli.resolve_profile(100_000_000, 2_000_000_000)
        worktree = Path("/tmp/small-llm-sft-worktree")
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
            mock.patch.object(sft_scaled_runtime.base, "_wandb_preflight"),
            mock.patch.object(sft_scaled_runtime.base, "_run", side_effect=capture_run),
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
        python_index = command.index("python")
        self.assertLess(command.index("torch==2.10.0"), python_index)
        self.assertLess(command.index("triton==3.6.0"), python_index)
        self.assertLess(command.index("fla-core==0.5.2"), python_index)
        self.assertLess(
            command.index("https://download.pytorch.org/whl/cu128"),
            python_index,
        )
        self.assertEqual(captured["cwd"], worktree)

    def test_variable_sft_rows_partition_without_duplication(self) -> None:
        for count in range(1, 18):
            left = dual_t4_sft._rank_row_indices(count, 0, 2)
            right = dual_t4_sft._rank_row_indices(count, 1, 2)
            self.assertEqual(sorted((*left, *right)), list(range(count)))
            self.assertTrue(set(left).isdisjoint(right))
            self.assertLessEqual(abs(len(left) - len(right)), 1)


if __name__ == "__main__":
    unittest.main()
