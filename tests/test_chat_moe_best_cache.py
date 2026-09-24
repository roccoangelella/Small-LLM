"""MoE chat follows the live best pointer without losing its old verified cache."""

import pytest

import chat


@pytest.fixture
def best_chat(monkeypatch, tmp_path):
    monkeypatch.setattr(chat, "_CHAT_MODEL_CACHE_DIR", tmp_path / "chat_models")
    pointer = {
        "checkpoint_id": "step-00075000",
        "prefix": "run/moe-100b-superbpe-003/checkpoints/step-00075000/last",
        "bucket_id": "owner/bucket",
        "best_pointer_sha256": "digest-one",
    }
    calls = []

    def remote(**kwargs):
        calls.append(("pointer", pointer["checkpoint_id"]))
        return dict(pointer)

    def download(**kwargs):
        calls.append(("download", kwargs["pointer_name"]))
        assert kwargs["repo_id"] == "owner/bucket"
        assert kwargs["run_id"] == chat._DEFAULT_MOE_RUN_ID
        root = kwargs["destination"] / pointer["checkpoint_id"]
        root.mkdir()
        info = {key: value for key, value in pointer.items() if key != "best_pointer_sha256"}
        return root, {**info, "pointer": "best"}

    def load(root, *, device, stage, run_id, allow_incomplete):
        calls.append(("load", root.name))
        assert root.is_dir()
        return root.name, "config", 1, None

    monkeypatch.setattr(chat, "_remote_moe_best", remote)
    monkeypatch.setattr("trainer.bucket_checkpoint_eval.download_verified_checkpoint", download)
    monkeypatch.setattr(chat, "_load_completed_checkpoint", load)
    return pointer, calls


def _get_best():
    return chat._download_model(
        repo_id="owner/repo",
        run_id=chat._DEFAULT_MOE_RUN_ID,
        source=chat._SOURCE_STORAGE_BUCKET,
        stage=chat._STAGE_PRETRAINED,
        device="cpu",
        track_best=True,
    )


def test_moe_best_cache_checks_remote_every_time_and_refreshes(best_chat):
    pointer, calls = best_chat
    first = _get_best()
    assert first[4]["cache_status"] == "downloaded"
    assert first[4]["pointer"] == "best"
    old_root = first[5]
    assert (old_root / "step-00075000").is_dir()

    second = _get_best()
    assert second[4]["cache_status"] == "hit"
    assert calls.count(("pointer", "step-00075000")) == 3
    assert calls.count(("download", "best")) == 1

    pointer.update(
        checkpoint_id="step-00077500",
        prefix="run/moe-100b-superbpe-003/checkpoints/step-00077500/last",
        best_pointer_sha256="digest-two",
    )
    third = _get_best()
    assert third[4]["checkpoint_id"] == "step-00077500"
    assert third[4]["cache_status"] == "downloaded"
    assert not (old_root / "step-00075000").exists()
    assert (old_root / "step-00077500").is_dir()
    assert _get_best()[4]["cache_status"] == "hit"
    # A mutated pointer under the same checkpoint ID must not count as a cache hit.
    pointer["best_pointer_sha256"] = "digest-three"
    assert _get_best()[4]["cache_status"] == "downloaded"


def test_moe_best_refresh_failure_preserves_previous_cache(best_chat, monkeypatch):
    pointer, _ = best_chat
    old_root = _get_best()[5]
    pointer.update(
        checkpoint_id="step-00077500",
        prefix="run/moe-100b-superbpe-003/checkpoints/step-00077500/last",
    )

    def broken(**kwargs):
        root = kwargs["destination"] / pointer["checkpoint_id"]
        root.mkdir()
        raise RuntimeError("manifest mismatch")

    monkeypatch.setattr("trainer.bucket_checkpoint_eval.download_verified_checkpoint", broken)
    with pytest.raises(RuntimeError, match="manifest mismatch"):
        _get_best()
    assert (old_root / "step-00075000").is_dir()
    cached = chat._read_cached_checkpoint(
        old_root,
        repo_id="owner/repo",
        run_id=chat._DEFAULT_MOE_RUN_ID,
        source=chat._SOURCE_STORAGE_BUCKET,
        stage=chat._STAGE_PRETRAINED,
    )
    assert cached[1]["checkpoint_id"] == "step-00075000"
    assert not list(old_root.parent.glob(".moe-100b-superbpe-003-new-*"))


def test_moe_best_pointer_failure_does_not_use_stale_cache(best_chat, monkeypatch):
    _get_best()
    def offline(**kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(chat, "_remote_moe_best", offline)
    with pytest.raises(RuntimeError, match="offline"):
        _get_best()


def test_moe_best_rejects_pointer_race(best_chat, monkeypatch):
    pointer, _ = best_chat
    root = _get_best()[5]
    pointer.update(
        checkpoint_id="step-00077500",
        prefix="run/moe-100b-superbpe-003/checkpoints/step-00077500/last",
    )

    def raced(**kwargs):
        checkpoint = kwargs["destination"] / "step-00080000"
        checkpoint.mkdir()
        return checkpoint, {
            "checkpoint_id": "step-00080000",
            "prefix": "wrong",
            "bucket_id": "owner/bucket",
            "pointer": "best",
        }

    monkeypatch.setattr("trainer.bucket_checkpoint_eval.download_verified_checkpoint", raced)
    with pytest.raises(RuntimeError, match="pointer changed"):
        _get_best()
    assert (root / "step-00075000").is_dir()


def test_moe_best_rejects_pointer_change_during_download(best_chat, monkeypatch):
    pointer, _ = best_chat
    from trainer.bucket_checkpoint_eval import download_verified_checkpoint as original

    def raced(**kwargs):
        result = original(**kwargs)
        pointer["best_pointer_sha256"] = "changed-mid-download"
        return result

    monkeypatch.setattr("trainer.bucket_checkpoint_eval.download_verified_checkpoint", raced)
    with pytest.raises(RuntimeError, match="pointer changed"):
        _get_best()
    assert not chat._chat_model_cache_dir(
        run_id=chat._DEFAULT_MOE_RUN_ID, stage=chat._STAGE_PRETRAINED
    ).exists()


def test_remote_best_validates_bucket_pointer(monkeypatch):
    from dataset.src.hf_bucket_checkpoint import HuggingFaceBucketCheckpointStore

    class Store:
        def __init__(self, bucket_id, *, token, private):
            assert bucket_id == "owner/bucket"
            assert token == "secret"
            assert private

        def read_json(self, path):
            assert path == "run/moe-100b-superbpe-003/best.json"
            return {
                "checkpoint_id": "step-00077500",
                "best_prefix": f"run/{chat._DEFAULT_MOE_RUN_ID}/checkpoints/step-00077500/last",
            }

    monkeypatch.setenv("SMALL_LLM_HF_CHECKPOINT_BUCKET_ID", "owner/bucket")
    monkeypatch.setattr(HuggingFaceBucketCheckpointStore, "__init__", Store.__init__)
    monkeypatch.setattr(HuggingFaceBucketCheckpointStore, "read_json", Store.read_json)
    info = chat._remote_moe_best(
        repo_id="owner/repo", run_id=chat._DEFAULT_MOE_RUN_ID, token="secret"
    )
    assert info["checkpoint_id"] == "step-00077500"
    assert len(info["best_pointer_sha256"]) == 64


def test_moe_snapshot_tokenizer_stays_available():
    assert chat._moe_chat_tokenizer(chat._DEFAULT_MOE_RUN_ID) == chat._moe_chat_tokenizer(
        chat._MOE_SNAPSHOT_RUN_ID
    )
