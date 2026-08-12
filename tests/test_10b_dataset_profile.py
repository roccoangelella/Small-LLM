"""Frozen geometry and producer transport tests for the 100M/10B dataset."""

from __future__ import annotations

from dataset.incremental_frontier import build_run_contract
from dataset.qualification import get_profile, production_arguments


def test_modal_10b_profile_freezes_block64_one_gib_hf_shards() -> None:
    profile = get_profile("modal-10b-b64")

    assert profile.run_id == "modal-10b-b64-dataset-001"
    assert profile.target_source_tokens == 10_000_000_000
    assert profile.minimum_source_tokens == 9_000_000_000
    assert profile.maximum_source_tokens == 11_000_000_000
    assert profile.checkpoint_source_tokens == 500_000_000
    assert profile.context_length == 2048
    assert profile.sequences_per_block == 64
    assert profile.target_shard_bytes == 1024**3
    assert profile.evict_remote_shards is True
    assert profile.incremental_frontier is True
    assert profile.nominal_training_tokens == 10_000_000_000
    assert profile.training_validation_blocks == 16

    block_bytes = (2048 + 1) * 64 * 2
    assert (1024**3) // block_bytes == 4094
    assert 4094 * block_bytes == 1_073_741_568
    assert 1024**3 - 4094 * block_bytes == 256


def test_modal_10b_profile_forces_incremental_remote_eviction() -> None:
    args = production_arguments("modal-10b-b64", ["--weights-file", "weights.json"])

    assert args[args.index("--run-id") + 1] == "modal-10b-b64-dataset-001"
    assert args[args.index("--target-shard-bytes") + 1] == str(1024**3)
    assert args[args.index("--sequences-per-block") + 1] == "64"
    assert args[args.index("--nominal-training-tokens") + 1] == "10000000000"
    assert args[args.index("--training-validation-blocks") + 1] == "16"
    assert "--remote-backend" not in args
    assert "--evict-remote-shards" in args
    assert "--incremental-frontier" in args


def test_modal_10b_contract_has_exact_prelaunch_horizon() -> None:
    profile = get_profile("modal-10b-b64")
    assert profile.run_id is not None and profile.nominal_training_tokens is not None
    contract = build_run_contract(
        run_id=profile.run_id,
        nominal_training_tokens=profile.nominal_training_tokens,
        target_source_tokens=profile.target_source_tokens,
        minimum_source_tokens=profile.minimum_source_tokens,
        maximum_source_tokens=profile.maximum_source_tokens,
        checkpoint_source_tokens=profile.checkpoint_source_tokens,
        context_length=profile.context_length,
        sequences_per_block=profile.sequences_per_block,
        target_shard_bytes=profile.target_shard_bytes,
        configuration_hash="a" * 64,
        schema_hash="b" * 64,
        work_plan_hash="c" * 64,
        validation_blocks=profile.training_validation_blocks,
    )
    assert contract["planned_train_blocks"] == 76_294
    assert contract["planned_train_target_tokens"] == 10_000_007_168
