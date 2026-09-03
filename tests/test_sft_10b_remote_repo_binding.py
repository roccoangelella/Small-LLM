from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
KAGGLE_SRC = ROOT / "kaggle" / "src"
if str(KAGGLE_SRC) not in sys.path:
    sys.path.insert(0, str(KAGGLE_SRC))

import dual_t4_sft_10b_same_data  # noqa: E402


class SFT10BRemoteRepoBindingTests(unittest.TestCase):
    def test_canonicalizes_stale_20m_remote_repos_to_100m_remotes(self) -> None:
        supplied = [
            "--parent-repo-id",
            "roccoangelella/small-llm-20m-qualification",
            "--checkpoint-repo-id",
            "roccoangelella/small-llm-20m-qualification",
            "--parent-run-id",
            dual_t4_sft_10b_same_data.EXPECTED_PARENT_RUN_ID,
        ]

        result = dual_t4_sft_10b_same_data._canonicalize_remote_repositories(supplied)

        parent_index = result.index("--parent-repo-id")
        checkpoint_index = result.index("--checkpoint-repo-id")
        self.assertEqual(
            result[parent_index + 1],
            dual_t4_sft_10b_same_data.EXPECTED_PARENT_REPO_ID,
        )
        self.assertEqual(
            result[checkpoint_index + 1],
            dual_t4_sft_10b_same_data.EXPECTED_CHECKPOINT_REPO_ID,
        )
        self.assertEqual(result.count("--parent-repo-id"), 1)
        self.assertEqual(result.count("--checkpoint-repo-id"), 1)

    def test_adds_missing_remote_repos_as_100m_remotes(self) -> None:
        result = dual_t4_sft_10b_same_data._canonicalize_remote_repositories([])

        self.assertEqual(
            result,
            [
                "--parent-repo-id",
                dual_t4_sft_10b_same_data.EXPECTED_PARENT_REPO_ID,
                "--checkpoint-repo-id",
                dual_t4_sft_10b_same_data.EXPECTED_CHECKPOINT_REPO_ID,
            ],
        )


if __name__ == "__main__":
    unittest.main()
