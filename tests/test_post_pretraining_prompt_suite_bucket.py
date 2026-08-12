"""Tests for Modal HF Storage Bucket post-pretraining evaluation."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from trainer.post_pretraining_prompt_suite_bucket import (
    _resolve_bucket_id,
    download_verified_bucket_checkpoint,
)


class PostPretrainingPromptSuiteBucketTests(unittest.TestCase):
    def test_bucket_id_uses_override_then_repo_derived_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                _resolve_bucket_id("owner/checkpoints"),
                "owner/checkpoints-checkpoints",
            )
        with patch.dict(
            os.environ,
            {"SMALL_LLM_HF_CHECKPOINT_BUCKET_ID": " owner/modal-bucket "},
            clear=True,
        ):
            self.assertEqual(_resolve_bucket_id("owner/checkpoints"), "owner/modal-bucket")

    def test_bucket_rejects_best_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "latest-only"):
                download_verified_bucket_checkpoint(
                    repo_id="owner/checkpoints",
                    run_id="100m-2b-data-001",
                    token="secret",
                    revision=None,
                    pointer_name="best",
                    destination=Path(temporary),
                )

    @patch("trainer.post_pretraining_prompt_suite_bucket._verify_published_checkpoint_manifest")
    @patch("trainer.post_pretraining_prompt_suite_bucket.verify_local_manifest")
    @patch("trainer.post_pretraining_prompt_suite_bucket.HuggingFaceBucketCheckpointStore")
    def test_latest_download_uses_modal_bucket_layout(
        self,
        store_type,
        verify_local,
        verify_published,
    ) -> None:
        pointer = {
            "checkpoint_id": "step-00015267",
            "last_prefix": "run/100m-2b-data-001/checkpoints/step-00015267/last",
            "checkpoint_manifest": {"version": 1, "files": []},
        }
        store = store_type.return_value
        store.read_json.return_value = pointer

        def materialize(_prefix: str, destination: Path) -> None:
            (destination / "checkpoint_manifest.json").write_text(
                '{"version": 1, "files": []}',
                encoding="utf-8",
            )

        store.download_tree.side_effect = materialize

        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(os.environ, {}, clear=True):
                checkpoint_root, info = download_verified_bucket_checkpoint(
                    repo_id="owner/checkpoints",
                    run_id="100m-2b-data-001",
                    token="secret",
                    revision=None,
                    pointer_name="latest",
                    destination=Path(temporary),
                )

        store_type.assert_called_once_with(
            "owner/checkpoints-checkpoints",
            token="secret",
            private=True,
        )
        store.read_json.assert_called_once_with("run/100m-2b-data-001/latest.json")
        store.download_tree.assert_called_once_with(
            "run/100m-2b-data-001/checkpoints/step-00015267/last",
            checkpoint_root,
        )
        verify_local.assert_called_once_with(checkpoint_root)
        verify_published.assert_called_once_with(
            checkpoint_root,
            pointer["checkpoint_manifest"],
        )
        self.assertEqual(info["checkpoint_source"], "hf_storage_bucket")
        self.assertEqual(info["checkpoint_id"], "step-00015267")
        self.assertEqual(info["pointer"], "latest")


if __name__ == "__main__":
    unittest.main()
