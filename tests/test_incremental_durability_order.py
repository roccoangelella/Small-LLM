"""Source-level guards for incremental producer crash ordering."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_ready_publication_follows_durable_progress_hook_and_precedes_eviction() -> None:
    source = (ROOT / "dataset" / "production" / "incremental_builder.py").read_text(
        encoding="utf-8"
    )
    function_start = source.index("def durable_state")
    function_end = source.index("\n        try:", function_start)
    body = source[function_start:function_end]

    write_progress = body.index("write_json_atomic(progress_path, state)")
    durable_commit = body.index("commit_progress_visibility()")
    publish_ready = body.index("publish_frontier(")
    evict = body.index("_evict_verified_local_shards(")
    assert write_progress < durable_commit < publish_ready < evict


def test_modal_producer_binds_volume_commit_to_durability_hook() -> None:
    source = (ROOT / "modal" / "rolling_producer.py").read_text(encoding="utf-8")
    assert "durable_progress_hook=commit_cache_volume" in source
    assert "periodic" not in source.lower()
