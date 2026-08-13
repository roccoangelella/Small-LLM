"""Trainer-process guard for the VPS-fed Beam dataset transport.

Activated only when ``SMALL_LLM_DATASET_REQUIRE_PRESEEDED=1``. The HF frontier
still supplies immutable shard metadata, but shard bytes must already be visible
in the mounted Beam Volume. No HF byte-download fallback is allowed.
"""
from __future__ import annotations

import os
import time


if os.environ.get("SMALL_LLM_DATASET_REQUIRE_PRESEEDED") == "1":
    import dataset.incremental_frontier as frontier

    try:
        _WAIT_SECONDS = float(os.environ.get("SMALL_LLM_DATASET_PRESEED_WAIT_SECONDS", "120"))
    except ValueError as error:
        raise RuntimeError("SMALL_LLM_DATASET_PRESEED_WAIT_SECONDS must be numeric") from error
    if _WAIT_SECONDS <= 0:
        raise RuntimeError("SMALL_LLM_DATASET_PRESEED_WAIT_SECONDS must be positive")

    def _preseeded_only(store, *, run_id, root, shard):
        del store, run_id
        destination = root / shard.filename
        deadline = time.monotonic() + _WAIT_SECONDS
        while True:
            if frontier._file_matches(root, shard):
                return destination
            if destination.is_symlink() or destination.is_dir():
                raise RuntimeError(
                    f"VPS-fed dataset shard path is unsafe in Beam Volume: {shard.filename}"
                )
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "VPS-fed Beam shard did not become visible before the bounded wait: "
                    f"{shard.filename}; keep the VPS producer ahead, then resume training"
                )
            time.sleep(1.0)

    frontier._download_verified = _preseeded_only
