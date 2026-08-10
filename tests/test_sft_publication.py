from __future__ import annotations

import unittest

from kaggle.sft_publish import _state_matches


class SFTPublicationTests(unittest.TestCase):
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
