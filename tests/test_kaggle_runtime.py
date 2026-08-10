"""Network-free regression tests for the profile-driven Kaggle runtime."""
from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / "kaggle"
if str(KAGGLE) not in sys.path:
    sys.path.insert(0, str(KAGGLE))

import runtime  # noqa: E402


class KaggleRuntimeTests(unittest.TestCase):
    def test_registered_profiles_preserve_qualified_identities(self) -> None:
        expected = {
            "100M": (
                "20m-100m-dataset-001",
                "20m-100m-data-004",
                "dataset.qualification_100m",
                "dataset.qualification_100m_report",
            ),
            "500M": (
                "20m-500m-dataset-001",
                "20m-500m-data-001",
                "dataset.qualification_500m",
                "dataset.qualification_500m_report",
            ),
            "2B": (
                "20m-2b-dataset-001",
                "20m-2b-data-001",
                "dataset.qualification_2b",
                "dataset.qualification_2b_report",
            ),
        }
        for profile in runtime.PROFILES.values():
            self.assertRegex(profile.launch_commit, r"^[0-9a-f]{40}$")
            self.assertEqual(profile.durability_every, 250)
            self.assertEqual(
                (
                    profile.dataset_run_id,
                    profile.wandb_run_id,
                    profile.qualification_module,
                    profile.qualification_report_module,
                ),
                expected[profile.token_label],
            )

    def test_2b_production_identity_is_fixed(self) -> None:
        profile = runtime.resolve_profile(20_000_000, 2_000_000_000)
        self.assertEqual(
            profile.production_identity(),
            {
                "run_id": "20m-2b-dataset-001",
                "target_source_tokens": 2_000_000_000,
                "minimum_source_tokens": 1_800_000_000,
                "maximum_source_tokens": 2_200_000_000,
                "checkpoint_source_tokens": 80_000_000,
                "target_reached": True,
                "remote_required": True,
            },
        )
        self.assertFalse(profile.run_microbatch_probe)
        self.assertEqual(profile.selected_microbatch, 4)

    def test_publication_bootstrap_replaces_shell_wrapper_behavior(self) -> None:
        command = runtime.publication_bootstrap_command(
            ["publish", "--model", "20M", "--tokens", "2B"],
            uv="/usr/bin/uv",
        )
        self.assertEqual(command[0], "/usr/bin/uv")
        self.assertIn("--env-file", command)
        self.assertIn(str(ROOT / ".env"), command)
        self.assertIn("--with-requirements", command)
        self.assertIn(str(KAGGLE / "requirements-100m-publish.txt"), command)
        self.assertEqual(
            command[-5:],
            ["publish", "--model", "20M", "--tokens", "2B"],
        )

    def test_top_level_kaggle_transport_archive_is_not_dataset_content(self) -> None:
        fake = SimpleNamespace(
            sha256=lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifest.json").write_bytes(b"manifest")
            baseline = runtime._dataset_tree_identity(fake, root)
            (root / "1.archive").write_bytes(b"transport")
            self.assertEqual(runtime._dataset_tree_identity(fake, root), baseline)
            (root / "train").mkdir()
            nested = root / "train" / "1.archive"
            nested.write_bytes(b"payload")
            first = runtime._dataset_tree_identity(fake, root)
            nested.write_bytes(b"changed")
            self.assertNotEqual(runtime._dataset_tree_identity(fake, root), first)

    def test_profile_handle_namespaces_are_distinct(self) -> None:
        self.assertEqual(
            runtime.PROFILES[(20_000_000, 100_000_000)].handle_env,
            "SMALL_LLM_KAGGLE_DATASET_HANDLE",
        )
        self.assertEqual(
            runtime.PROFILES[(20_000_000, 500_000_000)].handle_env,
            "SMALL_LLM_500M_KAGGLE_DATASET_HANDLE",
        )
        self.assertEqual(
            runtime.PROFILES[(20_000_000, 2_000_000_000)].handle_env,
            "SMALL_LLM_2B_KAGGLE_DATASET_HANDLE",
        )


if __name__ == "__main__":
    unittest.main()
