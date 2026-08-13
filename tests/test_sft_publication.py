from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kaggle.sft_publish import (
    PublishFailure,
    _remove_kagglehub_transport_artifacts,
    _state_matches,
)


class SFTPublicationTests(unittest.TestCase):
    def test_kagglehub_transport_artifacts_are_removed_before_identity_check(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "1.archive"
            marker = root / ".complete/datasets/owner/dataset/1/bundle.complete"
            payload = root / "train/payload.complete"
            archive.write_bytes(b"transport")
            marker.parent.mkdir(parents=True)
            marker.touch()
            payload.parent.mkdir()
            payload.write_bytes(b"dataset content")

            removed = _remove_kagglehub_transport_artifacts(root)

            self.assertEqual(
                removed,
                (
                    "1.archive",
                    ".complete/datasets/owner/dataset/1/bundle.complete",
                ),
            )
            self.assertFalse(archive.exists())
            self.assertFalse((root / ".complete").exists())
            self.assertEqual(payload.read_bytes(), b"dataset content")

    def test_unexpected_completion_marker_content_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / ".complete/datasets/owner/dataset/1/bundle.complete"
            marker.parent.mkdir(parents=True)
            marker.write_bytes(b"unexpected")

            with self.assertRaisesRegex(
                PublishFailure,
                "unexpected KaggleHub completion-marker content",
            ):
                _remove_kagglehub_transport_artifacts(root)
            self.assertTrue(marker.is_file())

    def test_publication_state_matches_exact_bundle_identity(self) -> None:
        identity = {
            "tree_sha256": "a" * 64,
            "file_count": 12,
            "total_bytes": 3456,
        }
        state = {
            "status": "upload_submitted",
            "handle": "owner/private-sft",
            "bundle_manifest_sha256": "b" * 64,
            **identity,
        }
        self.assertTrue(
            _state_matches(
                state,
                handle="owner/private-sft",
                identity=identity,
                bundle_manifest_sha256="b" * 64,
            )
        )
        self.assertFalse(
            _state_matches(
                state,
                handle="owner/private-sft",
                identity={**identity, "tree_sha256": "c" * 64},
                bundle_manifest_sha256="b" * 64,
            )
        )
        self.assertFalse(
            _state_matches(
                state,
                handle="owner/other",
                identity=identity,
                bundle_manifest_sha256="b" * 64,
            )
        )
        self.assertFalse(
            _state_matches(
                state,
                handle="owner/private-sft",
                identity=identity,
                bundle_manifest_sha256="d" * 64,
            )
        )


if __name__ == "__main__":
    unittest.main()
