#!/usr/bin/env python3
"""Inspect whether the exact 100M/10B step-12,500 probe source still exists.

Usage from a clean repository root:

    python beam/inspect_decay_probe_source.py

This is CPU-only. It inspects the Beam run Volume and the configured Hugging
Face model-repository live pointer; it never allocates a GPU and never mutates
checkpoint state.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beam import function  # noqa: E402
from beam import launch as base  # noqa: E402

SOURCE_RUN_ID = "100m-10b-data-001"
SOURCE_STEP = 12_500
SOURCE_CHECKPOINT_ID = f"step-{SOURCE_STEP:08d}"
_CHECKPOINT = re.compile(r"^step-(\d{8})$")


def _checkpoint_steps(checkpoint_dir: Path) -> list[int]:
    result: list[int] = []
    if not checkpoint_dir.is_dir():
        return result
    for path in checkpoint_dir.iterdir():
        match = _CHECKPOINT.fullmatch(path.name) if path.is_dir() else None
        if match is not None:
            result.append(int(match.group(1)))
    return sorted(result)


def _nearest(steps: list[int]) -> dict[str, int | None]:
    before = [step for step in steps if step <= SOURCE_STEP]
    after = [step for step in steps if step >= SOURCE_STEP]
    return {
        "nearest_at_or_before": max(before) if before else None,
        "nearest_at_or_after": min(after) if after else None,
    }


@function(
    name="small-llm-inspect-decay-probe-source",
    image=base.CPU_IMAGE,
    cpu=2,
    memory="4Gi",
    timeout=300,
    retries=1,
    secrets=base.SECRETS,
    volumes=[base.RUN_VOLUME],
    env=base.RUNTIME_ENV,
)
def inspect_remote() -> dict[str, object]:
    base._install_beam_imports()
    import model_repo_checkpoint as transport
    import runtime as runtime_base
    from dataset.src.joint_checkpoint import verify_local_manifest

    transport.install_model_repo_checkpoint_transport()

    checkpoint_dir = base.RUN_ROOT / SOURCE_RUN_ID / "checkpoints"
    steps = _checkpoint_steps(checkpoint_dir)
    exact = checkpoint_dir / SOURCE_CHECKPOINT_ID
    exact_local_valid = False
    exact_local_error: str | None = None
    if exact.is_dir():
        try:
            verify_local_manifest(exact)
        except Exception as error:  # noqa: BLE001 - diagnostics only
            exact_local_error = f"{type(error).__name__}: {error}"
        else:
            exact_local_valid = True

    repo_id = runtime_base._hf_model_repo_id()
    store = runtime_base._hf_model_repo_store()
    pointer = store.read_json(f"run/{SOURCE_RUN_ID}/latest.json")
    if pointer is not None and not isinstance(pointer, Mapping):
        raise RuntimeError("Hugging Face live checkpoint pointer is not an object")
    hf_latest = None if pointer is None else pointer.get("checkpoint_id")

    # The production transport is rolling latest-only and super-squashes model
    # repository history. Listing current files tells us whether an old exact
    # checkpoint directory somehow survived despite that policy.
    prefix = f"run/{SOURCE_RUN_ID}/checkpoints/"
    files = store.api.list_repo_files(
        repo_id=store.repo_id,
        repo_type=store.repo_type,
        revision=store.revision or "main",
        token=store.token,
    )
    remote_ids: set[str] = set()
    for name in files:
        if not isinstance(name, str) or not name.startswith(prefix):
            continue
        remainder = name[len(prefix):]
        candidate = remainder.split("/", 1)[0]
        if _CHECKPOINT.fullmatch(candidate):
            remote_ids.add(candidate)
    remote_steps = sorted(int(item.removeprefix("step-")) for item in remote_ids)

    return {
        "source_run_id": SOURCE_RUN_ID,
        "requested_checkpoint_id": SOURCE_CHECKPOINT_ID,
        "beam_checkpoint_dir": str(checkpoint_dir),
        "beam_exact_present": exact.is_dir(),
        "beam_exact_valid": exact_local_valid,
        "beam_exact_error": exact_local_error,
        "beam_checkpoint_count": len(steps),
        "beam_first_step": steps[0] if steps else None,
        "beam_last_step": steps[-1] if steps else None,
        "beam_nearest": _nearest(steps),
        "beam_recent_steps": steps[-12:],
        "hf_repo_id": repo_id,
        "hf_latest_checkpoint_id": hf_latest,
        "hf_exact_currently_present": SOURCE_CHECKPOINT_ID in remote_ids,
        "hf_checkpoint_count_current_tree": len(remote_steps),
        "hf_nearest": _nearest(remote_steps),
        "hf_current_checkpoint_ids": sorted(remote_ids),
        "retention": "rolling latest-only model-repo history squashed",
    }


def main() -> int:
    result = base._require_remote_mapping(
        inspect_remote.remote(),
        label="decay probe source inspection",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
