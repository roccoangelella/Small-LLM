"""CPU-only production streaming, resume identity, and frontier status contracts."""
from __future__ import annotations

import copy
import io
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

import moe_production as production
from dataset.incremental_frontier import (
    SHARD_FRONTIER_FILENAME, build_run_contract, publish_frontier, publish_run_contract,
)
from dataset.incremental_stage import stage_incremental_window_when_ready
from trainer.cli_setup import _rolling_cache
from trainer.config import TrainerConfig
from trainer.identity import checkpoint_identity
from trainer.shards import SchemaV2ShardReader
from MOE_model.__main__ import parse_args
from MOE_model.config import MoEModelConfig
from tests.test_incremental_frontier import FakeStore, _entry
from tests.test_production_checkpoint_sequence import _write_checkpoint


@pytest.fixture
def corpus(tmp_path):
    store = FakeStore()
    # Exercise the existing-public-bucket case: consumption must not create it.
    store.ensure_bucket = Mock(side_effect=AssertionError("consumer cannot create a bucket"))
    contract = build_run_contract(
        run_id="dataset-001", nominal_training_tokens=204_800,
        target_source_tokens=204_800, minimum_source_tokens=200_000,
        maximum_source_tokens=210_000, checkpoint_source_tokens=20_480,
        context_length=2048, sequences_per_block=1, target_shard_bytes=4098,
        configuration_hash="a" * 64, schema_hash="b" * 64, work_plan_hash="c" * 64,
        validation_blocks=1,
    )
    publish_run_contract(store, run_id="dataset-001", contract=contract)
    payload = b"\x01\x00" * 2049
    rows = []
    for split, index in (("train", 0), ("train", 1), ("validation", 0)):
        row = _entry(split, index, index, index, payload)
        row.update(sequence_count=1, context_length=2048)
        store.blobs[store.object_key("dataset-001", row["filename"])] = payload
        rows.append(row)
    publish_frontier(store, run_id="dataset-001", contract=contract,
                     durability_manifest={"shards": rows})
    request = production.ProductionRequest(
        run_id="moe-stream", dataset_dir=str(tmp_path / "dataset"), total_steps=100,
        precision="bf16", microbatch_size=1, source_commit="a" * 40,
        dataset_shard_bucket="public/corpus", dataset_shard_run_id="dataset-001",
    )
    return store, contract, rows, request


def test_streaming_command_and_static_defaults(tmp_path, corpus):
    _, _, _, request = corpus
    command = production.build_training_command(
        request, run_root=tmp_path, schedule=production.DISPLAY_SCHEDULE,
    )
    args = parse_args(command[3:] + ["--device", "cpu"])
    assert args.dataset_shard_bucket == "public/corpus"
    assert args.dataset_shard_run_id == "dataset-001"
    assert args.dataset_manifest == Path(request.dataset_dir) / "manifest.json"
    assert args.dataset_shard_token_env == "HF_TOKEN"
    assert args.dataset_shard_prefetch == 1
    assert args.dataset_shard_wait_timeout_seconds == 10_800
    assert production.request_from_payload(production.request_payload(request)) == request
    static = replace(request, dataset_shard_bucket="", dataset_shard_run_id="")
    old_command = production.build_training_command(
        static, run_root=tmp_path, schedule=production.DISPLAY_SCHEDULE,
    )
    assert command[:len(old_command)] == old_command
    assert not any(flag.startswith("--dataset-shard") for flag in old_command)
    assert "--dataset-manifest" not in old_command
    assert parse_args(old_command[3:] + ["--device", "cpu"]).dataset_shard_wait_timeout_seconds == 0


@pytest.mark.parametrize("overrides", [dict(dataset_shard_bucket=""), dict(dataset_shard_run_id="")])
def test_streaming_fields_must_be_paired(corpus, overrides):
    with pytest.raises(ValueError, match="supplied together"):
        replace(corpus[3], **overrides)


def test_prepare_uses_existing_stager_without_completeness_gate(corpus):
    store, _, _, request = corpus
    with patch("moe_production._dataset_store", return_value=store), patch(
        "moe_production.stage_incremental_window_when_ready",
        side_effect=stage_incremental_window_when_ready,
    ) as stage:
        result = production.prepare_dataset(request)
    assert result["status"] == "ready"
    assert result["train_blocks"] == 100  # Frozen horizon, only two blocks are READY.
    assert result["validation_blocks"] == 1
    assert result["first_block_id"] == 0
    stage.assert_called_once_with(
        store=store, run_id="dataset-001", destination=Path(request.dataset_dir),
        start_block_id=0, ensure_bucket=False,
    )
    store.ensure_bucket.assert_not_called()
    assert production.resolve_schedule(request.dataset_dir)["steps"] == 100


def test_prepare_checks_contract_schedule_before_training(corpus):
    store, _, _, request = corpus
    with patch("moe_production._dataset_store", return_value=store):
        with pytest.raises(ValueError, match="trainer plan"):
            production.prepare_dataset(replace(request, total_steps=101))


def test_prepare_propagates_staging_failure(corpus):
    store, _, _, request = corpus
    with patch("moe_production._dataset_store", return_value=store), patch(
        "moe_production.stage_incremental_window_when_ready", side_effect=TimeoutError("lead buffer"),
    ), pytest.raises(TimeoutError, match="lead buffer"):
        production.prepare_dataset(request)


def test_identity_and_resume_survive_frontier_growth(tmp_path, corpus):
    store, contract, rows, request = corpus
    root = Path(request.dataset_dir)
    with patch("moe_production._dataset_store", return_value=store):
        production.prepare_dataset(request)
    original_manifest = (root / "manifest.json").read_bytes()
    identity_args = dict(model_config=MoEModelConfig.accepted(), trainer_config=TrainerConfig())
    original_identity = checkpoint_identity(root, **identity_args)
    source = SchemaV2ShardReader(root, semantic_vocab_size=8000)
    state = {**source.state_dict(), "last_consumed_block_id": 2}
    # The resumed cursor lies beyond the bootstrap manifest's blocks 0 and 1.
    for index in (2, 3):
        row = {**rows[0], "filename": f"train/train-{index:06d}.bin",
               "first_block_id": index, "last_block_id": index}
        store.blobs[store.object_key("dataset-001", row["filename"])] = b"\x01\x00" * 2049
        rows.append(row)
    publish_frontier(store, run_id="dataset-001", contract=contract,
                     durability_manifest={"shards": rows})
    # Cursor 2 requires blocks 2+3 in the staged window. Block 0 need not be on disk.
    _write_checkpoint(production.checkpoint_dir(request, run_root=tmp_path), 2)
    (root / "train/train-000000.bin").unlink()
    with patch("moe_production._dataset_store", return_value=store):
        prepared = production.prepare_dataset(request, run_root=tmp_path)
    assert prepared["first_block_id"] == 2
    assert (root / "manifest.json").read_bytes() == original_manifest
    assert checkpoint_identity(root, **identity_args) == original_identity
    # The production CLI actually constructs the dynamic cache, including its timeout.
    command = production.build_training_command(request, run_root=tmp_path)
    args = parse_args(command[3:] + ["--device", "cpu"])
    with patch.dict("os.environ", {"HF_TOKEN": "test"}), patch(
        "dataset.src.hf_bucket_shards.HuggingFaceBucketShardStore", return_value=store,
    ):
        cache = _rolling_cache(args)
    try:
        assert cache.wait_timeout_seconds == 10800
        resumed = SchemaV2ShardReader(root, semantic_vocab_size=8000, cache_manager=cache)
        # Avoid speculative waiting beyond this test's prefix when restoring at block 3.
        with patch.object(cache, "_prefetch_successor"):
            resumed.load_state_dict(state)
            assert resumed.next_batch().block_id == 3
    finally:
        cache.close()
    # Completing production changes control-plane state, never manifest identity.
    frontier = copy.deepcopy(store.json_objects[store.object_key("dataset-001", SHARD_FRONTIER_FILENAME)])
    frontier["producer_complete"] = True
    with patch("dataset.incremental_frontier.read_frontier", return_value=frontier):
        stage_incremental_window_when_ready(
            store=store, run_id="dataset-001", destination=root, start_block_id=2, ensure_bucket=False,
        )
    assert (root / "manifest.json").read_bytes() == original_manifest


@pytest.mark.parametrize("complete, expected", [(False, "waiting_for_corpus"), (True, "incomplete")])
def test_child_early_exit_uses_live_frontier(tmp_path, corpus, complete, expected):
    store, contract, _, request = corpus
    with patch("moe_production._dataset_store", return_value=store):
        production.prepare_dataset(request)
        frontier = {"run_id": "dataset-001", "contract_sha256": contract["contract_sha256"],
                    "last_ready_train_block_id": 28, "producer_complete": complete}
        with patch("moe_production.read_frontier", return_value=frontier):
            result = production.run_provider_payload(
                production.request_payload(request), run_root=tmp_path, repo_root=tmp_path,
                popen_factory=lambda *a, **kw: SimpleNamespace(stdout=io.StringIO(""), wait=lambda: 0),
            )
    assert result["status"] == expected
    assert result["frontier"]["last_ready_train_block_id"] == 28
    assert result["steps_reached"] == 0


@pytest.mark.parametrize("value", ["-1", "nan", "inf"])
def test_cli_rejects_invalid_wait_timeouts(value):
    with pytest.raises(SystemExit, match="finite and non-negative"):
        parse_args(["--dataset-dir", ".", "--checkpoint-dir", ".", "--steps", "1",
                    "--dataset-shard-wait-timeout-seconds", value])


def test_finished_streaming_run_does_not_stage_again(tmp_path, corpus):
    _, _, _, request = corpus
    _write_checkpoint(production.checkpoint_dir(request, run_root=tmp_path), request.total_steps)
    with patch("moe_production._dataset_store", side_effect=AssertionError("no bucket needed")):
        result = production.prepare_dataset(request, run_root=tmp_path)
    assert result["training_complete"] is True


def test_streaming_store_requires_existing_token_and_never_creates_bucket(corpus):
    request = corpus[3]
    with patch.dict("os.environ", {"HF_TOKEN": ""}), pytest.raises(RuntimeError, match="HF_TOKEN"):
        production._dataset_store(request)
    with patch.dict("os.environ", {"HF_TOKEN": "test"}), patch(
        "moe_production.HuggingFaceBucketShardStore",
    ) as store:
        production._dataset_store(request)
    store.assert_called_once_with("public/corpus", token="test", create_bucket=False)


def test_child_frontier_status_rejects_contract_drift(tmp_path, corpus):
    store, contract, _, request = corpus
    with patch("moe_production._dataset_store", return_value=store):
        production.prepare_dataset(request)
        with patch("moe_production.read_run_contract", return_value={**contract, "contract_sha256": "d" * 64}), \
             pytest.raises(RuntimeError, match="contract changed"):
            production.run_provider_payload(
                production.request_payload(request), run_root=tmp_path, repo_root=tmp_path,
                popen_factory=lambda *a, **kw: SimpleNamespace(stdout=io.StringIO(""), wait=lambda: 0),
            )
