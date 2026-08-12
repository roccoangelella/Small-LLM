"""Tests for canonical stable Hugging Face model artifacts."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trainer.model_artifact import download_verified_model_artifact, resolve_model_artifact
from trainer.post_pretraining_prompt_suite_model import download_verified_stable_model


class ModelArtifactTests(unittest.TestCase):
    def test_resolve_prefers_artifact_pointer(self) -> None:
        store = unittest.mock.Mock()
        store.read_json.return_value = {
            "run_id": "100m-2b-data-001",
            "checkpoint_id": "step-00015267",
            "huggingface_path": "models/100m-2b-data-001/step-00015267",
            "is_final": True,
        }
        checkpoint_id, prefix, metadata = resolve_model_artifact(
            store,
            run_id="100m-2b-data-001",
        )
        self.assertEqual(checkpoint_id, "step-00015267")
        self.assertEqual(prefix, "models/100m-2b-data-001/step-00015267")
        self.assertTrue(metadata["is_final"])

    def test_resolve_discovers_manually_moved_latest_step(self) -> None:
        store = unittest.mock.Mock()
        store.read_json.return_value = None
        store.api.list_repo_files.return_value = [
            "models/100m-2b-data-001/step-00015000/checkpoint.json",
            "models/100m-2b-data-001/step-00015267/checkpoint.json",
            "models/100m-2b-data-001/step-00015267/trainer_state.pkl",
        ]
        store._hub_kwargs.return_value = {"repo_id": "owner/models"}
        checkpoint_id, prefix, metadata = resolve_model_artifact(
            store,
            run_id="100m-2b-data-001",
        )
        self.assertEqual(checkpoint_id, "step-00015267")
        self.assertEqual(prefix, "models/100m-2b-data-001/step-00015267")
        self.assertTrue(metadata["pointer_discovered"])

    @patch("trainer.model_artifact.verify_local_manifest")
    @patch("trainer.model_artifact.HuggingFaceCheckpointStore")
    def test_download_accepts_native_stable_checkpoint_without_publisher_manifest(
        self,
        store_type,
        verify_local,
    ) -> None:
        """Stable models/... snapshots require local_manifest, not checkpoint_manifest."""

        store = store_type.return_value
        store.read_json.return_value = {
            "run_id": "100m-2b-data-001",
            "checkpoint_id": "step-00015267",
            "huggingface_path": "models/100m-2b-data-001/step-00015267",
            "is_final": True,
        }

        def materialize(_prefix: str, destination: Path) -> None:
            # Match the normalized completed checkpoint shape seen on HF. There
            # is intentionally no checkpoint_manifest.json: that file belongs to
            # the two-phase live run/... publication protocol.
            (destination / "checkpoint.json").write_text("{}", encoding="utf-8")
            (destination / "local_manifest.json").write_text('{"files": []}', encoding="utf-8")
            (destination / "trainer_state.pkl").write_bytes(b"native-checkpoint")

        store.download_tree.side_effect = materialize
        with tempfile.TemporaryDirectory() as temporary:
            root, info = download_verified_model_artifact(
                repo_id="owner/models",
                run_id="100m-2b-data-001",
                token="secret",
                revision=None,
                destination=Path(temporary),
            )

        store.download_tree.assert_called_once_with(
            "models/100m-2b-data-001/step-00015267",
            root,
        )
        verify_local.assert_called_once_with(root)
        self.assertFalse((root / "checkpoint_manifest.json").exists())
        self.assertEqual(info["checkpoint_source"], "hf_model_artifact")
        self.assertEqual(info["checkpoint_id"], "step-00015267")

    def test_stable_evaluator_rejects_best_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "--pointer latest"):
                download_verified_stable_model(
                    repo_id="owner/models",
                    run_id="100m-2b-data-001",
                    token="secret",
                    revision=None,
                    pointer_name="best",
                    destination=Path(temporary),
                )


if __name__ == "__main__":
    unittest.main()
