from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from post_training.sft.checkpoints import download_parent_checkpoint


def test_parent_download_falls_back_to_stable_model_artifact_when_live_pointer_missing(
    tmp_path: Path,
) -> None:
    stable_root = tmp_path / "step-00015267"
    stable_info = {
        "checkpoint_source": "hf_model_artifact",
        "checkpoint_id": "step-00015267",
    }

    with (
        patch(
            "post_training.sft.checkpoints.download_verified_checkpoint",
            side_effect=RuntimeError(
                "Hugging Face pointer is missing: run/100m-2b-data-001/best.json"
            ),
        ) as live,
        patch(
            "post_training.sft.checkpoints.download_verified_model_artifact",
            return_value=(stable_root, stable_info),
        ) as stable,
    ):
        root, info = download_parent_checkpoint(
            repo_id="owner/qualification",
            run_id="100m-2b-data-001",
            pointer="best",
            token="token",
            destination=tmp_path / "download",
        )

    assert root == stable_root
    assert info["checkpoint_source"] == "hf_model_artifact"
    assert info["requested_parent_pointer"] == "best"
    assert info["parent_resolution"] == "stable_model_artifact_fallback"
    live.assert_called_once()
    stable.assert_called_once()


def test_parent_download_does_not_mask_live_integrity_failures(tmp_path: Path) -> None:
    with (
        patch(
            "post_training.sft.checkpoints.download_verified_checkpoint",
            side_effect=RuntimeError("checkpoint manifest hash mismatch"),
        ),
        patch(
            "post_training.sft.checkpoints.download_verified_model_artifact"
        ) as stable,
    ):
        with pytest.raises(RuntimeError, match="checkpoint manifest hash mismatch"):
            download_parent_checkpoint(
                repo_id="owner/qualification",
                run_id="100m-2b-data-001",
                pointer="best",
                token="token",
                destination=tmp_path / "download",
            )

    stable.assert_not_called()


def test_parent_download_keeps_live_pointer_when_available(tmp_path: Path) -> None:
    live_root = tmp_path / "step-00000100"
    live_info = {"checkpoint_source": "hf_live_checkpoint"}

    with (
        patch(
            "post_training.sft.checkpoints.download_verified_checkpoint",
            return_value=(live_root, live_info),
        ),
        patch(
            "post_training.sft.checkpoints.download_verified_model_artifact"
        ) as stable,
    ):
        root, info = download_parent_checkpoint(
            repo_id="owner/qualification",
            run_id="20m-500m-data-001",
            pointer="best",
            token="token",
            destination=tmp_path / "download",
        )

    assert root == live_root
    assert info == live_info
    stable.assert_not_called()


def test_parent_download_can_use_verified_storage_bucket_latest(tmp_path: Path) -> None:
    bucket_root = tmp_path / "step-00076294"
    bucket_info = {
        "checkpoint_source": "hf_storage_bucket",
        "checkpoint_id": "step-00076294",
        "pointer": "latest",
    }

    with (
        patch(
            "trainer.post_pretraining_prompt_suite_bucket.download_verified_bucket_checkpoint",
            return_value=(bucket_root, bucket_info),
        ) as bucket,
        patch(
            "post_training.sft.checkpoints.download_verified_checkpoint"
        ) as live,
        patch(
            "post_training.sft.checkpoints.download_verified_model_artifact"
        ) as stable,
    ):
        root, info = download_parent_checkpoint(
            repo_id="roccoangelella/small-llm-100m-qualification",
            run_id="100m-10b-deep-decay-from-step15500",
            pointer="latest",
            transport="hf_storage_bucket",
            token="token",
            destination=tmp_path / "download",
        )

    assert root == bucket_root
    assert info["checkpoint_source"] == "hf_storage_bucket"
    assert info["checkpoint_id"] == "step-00076294"
    assert info["requested_parent_pointer"] == "latest"
    assert info["parent_resolution"] == "hf_storage_bucket"
    bucket.assert_called_once_with(
        repo_id="roccoangelella/small-llm-100m-qualification",
        run_id="100m-10b-deep-decay-from-step15500",
        token="token",
        revision=None,
        pointer_name="latest",
        destination=tmp_path / "download",
    )
    live.assert_not_called()
    stable.assert_not_called()


def test_parent_bucket_transport_does_not_allow_best_pointer(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="rolling latest-only retention"):
        download_parent_checkpoint(
            repo_id="owner/qualification",
            run_id="100m-10b-deep-decay-from-step15500",
            pointer="best",
            transport="hf_storage_bucket",
            token="token",
            destination=tmp_path / "download",
        )


def test_parent_download_rejects_unknown_transport(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="parent transport must be one of"):
        download_parent_checkpoint(
            repo_id="owner/qualification",
            run_id="100m-10b-deep-decay-from-step15500",
            pointer="latest",
            transport="made-up",
            token="token",
            destination=tmp_path / "download",
        )
