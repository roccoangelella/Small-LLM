"""Explicit opt-in smoke test for the real Drive and private Hub backends.

It is deliberately disabled in ordinary test runs.  The generated objects use
an isolated run ID and are retained for an operator to inspect or clean up.
"""

from __future__ import annotations

import os
import tempfile
import unittest
import uuid
from pathlib import Path

from dataset.src.remote import GoogleDriveShardStore, HuggingFaceCheckpointStore, sha256_path


class LiveRemoteSmokeTest(unittest.TestCase):
    def test_authenticated_upload_download_and_checksum(self) -> None:
        if os.environ.get("SMALL_LLM_LIVE_REMOTE_SMOKE") != "1":
            self.skipTest("set SMALL_LLM_LIVE_REMOTE_SMOKE=1 to contact real remote storage")
        folder_id = os.environ.get("SMALL_LLM_DRIVE_FOLDER_ID")
        repo_id = os.environ.get("SMALL_LLM_HF_REPO_ID")
        if not folder_id or not repo_id:
            self.fail("set SMALL_LLM_DRIVE_FOLDER_ID and SMALL_LLM_HF_REPO_ID for the live smoke test")

        run_id = f"smoke-{uuid.uuid4().hex}"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = root / "payload.bin"
            payload.write_bytes(b"small-llm remote smoke\x00\x01\x02")
            expected = sha256_path(payload)

            drive = GoogleDriveShardStore.from_credentials(folder_id=folder_id)
            uploaded = drive.upload_finalized_shard(
                run_id=run_id, logical_name="train/smoke.bin", local_path=payload,
            )
            drive.verify_remote_shard(
                run_id=run_id, logical_name="train/smoke.bin", file_id=str(uploaded["file_id"]),
                byte_size=payload.stat().st_size, sha256=expected,
            )
            downloaded = root / "downloaded.bin"
            drive.download_shard(
                run_id=run_id, logical_name="train/smoke.bin", file_id=str(uploaded["file_id"]),
                destination=downloaded, byte_size=payload.stat().st_size, sha256=expected,
            )
            self.assertEqual(sha256_path(downloaded), expected)

            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "payload.bin").write_bytes(payload.read_bytes())
            hub = HuggingFaceCheckpointStore(repo_id, token=os.environ.get("HF_TOKEN"), private=True)
            remote_prefix = f"small-llm-smoke/{run_id}"
            uploaded_hashes = hub.upload_tree(remote_prefix, checkpoint)
            self.assertEqual(uploaded_hashes[f"{remote_prefix}/payload.bin"], expected)
            restored = root / "restored"
            hub.download_tree(remote_prefix, restored)
            self.assertEqual(sha256_path(restored / "payload.bin"), expected)
