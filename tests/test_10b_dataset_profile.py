"""Frozen geometry and producer transport tests for the 100M/10B dataset."""

from __future__ import annotations

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

    block_bytes = (2048 + 1) * 64 * 2
    assert (1024**3) // block_bytes == 4094
    assert 4094 * block_bytes == 1_073_741_568
    assert 1024**3 - 4094 * block_bytes == 256


def test_modal_10b_profile_forces_remote_eviction_without_backend_switch() -> None:
    args = production_arguments("modal-10b-b64", ["--weights-file", "weights.json"])

    assert args[args.index("--run-id") + 1] == "modal-10b-b64-dataset-001"
    assert args[args.index("--target-shard-bytes") + 1] == str(1024**3)
    assert args[args.index("--sequences-per-block") + 1] == "64"
    assert "--remote-backend" not in args
    assert "--evict-remote-shards" in args
