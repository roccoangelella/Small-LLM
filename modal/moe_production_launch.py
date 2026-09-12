#!/usr/bin/env python3
"""Modal H100 launcher for the accepted production MoE identity."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

_SOURCE_FILE = Path(__file__).resolve()
_REMOTE_REPO_ROOT = Path("/root/small-llm")
# Modal copies the entrypoint to /root while the repository is mounted at /root/small-llm.
if _SOURCE_FILE.parent == Path("/root") and (
    _REMOTE_REPO_ROOT / "modal" / "launch.py"
).is_file():
    _REPO_ROOT = _REMOTE_REPO_ROOT
else:
    _REPO_ROOT = _SOURCE_FILE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import moe_production as _production


def _load_existing_launcher():
    source = _REPO_ROOT / "modal" / "launch.py"
    spec = importlib.util.spec_from_file_location("small_llm_modal_base_launch", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load canonical Modal launcher: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_base = _load_existing_launcher()
_modal = _base.modal
app = _modal.App("small-llm-moe-production", image=_base.IMAGE)

REMOTE_REPO = _base.REMOTE_REPO
DATA_ROOT = _base.DATA_ROOT
RUN_ROOT = _base.RUN_ROOT
CACHE_ROOT = _base.CACHE_ROOT
DATA_VOLUME = _base.DATA_VOLUME
RUN_VOLUME = _base.RUN_VOLUME
CACHE_VOLUME = _base.CACHE_VOLUME
TRAINING_SECRET = _base.TRAINING_SECRET


def _dataset_volume(request: _production.ProductionRequest):
    destination = Path(request.dataset_dir).resolve()
    for root, volume in ((DATA_ROOT, DATA_VOLUME), (CACHE_ROOT, CACHE_VOLUME), (RUN_ROOT, RUN_VOLUME)):
        if destination.is_relative_to(root):
            return volume
    raise ValueError("streaming dataset_dir must be inside a mounted data, cache or run volume")


def _require_source_commit(source_commit: str) -> None:
    actual = _base._local_source_commit()
    if source_commit != actual:
        raise RuntimeError(
            f"--source-commit {source_commit} does not match clean checkout HEAD {actual}"
        )


@app.function(
    cpu=2,
    memory=8192,
    timeout=24 * 60 * 60,
    retries=1,
    secrets=[TRAINING_SECRET],
    volumes={
        str(DATA_ROOT): DATA_VOLUME,
        str(RUN_ROOT): RUN_VOLUME,
        str(CACHE_ROOT): CACHE_VOLUME,
    },
)
def prepare_production_cpu(payload: dict[str, object]) -> dict[str, object]:
    """Decode a real training block with the accepted 8,000-token semantic bound."""

    request = _production.request_from_payload(payload)
    if request.streaming:
        volume = _dataset_volume(request)
        volume.reload()
    result = _production.prepare_dataset(request, run_root=RUN_ROOT)
    if request.streaming:
        volume.commit()
    return result


@app.function(
    gpu="H100",
    timeout=24 * 60 * 60,
    retries=3,
    single_use_containers=True,
    secrets=[TRAINING_SECRET],
    volumes={
        str(DATA_ROOT): DATA_VOLUME,
        str(RUN_ROOT): RUN_VOLUME,
        str(CACHE_ROOT): CACHE_VOLUME,
    },
)
def train_production_h100(payload: dict[str, object]) -> dict[str, object]:
    """Run only ``MoEModelConfig.accepted()`` on the production H100 path."""

    request = _production.request_from_payload(payload)
    if request.streaming:
        _dataset_volume(request).reload()
    return _production.run_provider_payload(
        payload,
        run_root=RUN_ROOT,
        repo_root=REMOTE_REPO,
        volume_commit=RUN_VOLUME.commit,
    )


@app.local_entrypoint()
def main(
    run_id: str,
    dataset_dir: str,
    steps: int,
    source_commit: str,
    precision: str = "bf16",
    microbatch_size: int = 64,
    resume: str = "",
    sequences_per_block: int = 0,
    checkpoint_every_steps: int = 1000,
    keep_last_checkpoints: int = 3,
    milestone_every_steps: int = 0,
    max_wall_seconds: float = 23 * 60 * 60,
    validation_blocks: int = -1,
    compile_mode: str = "blocks",
    allow_partial_corpus: bool = False,
    dataset_shard_bucket: str = "",
    dataset_shard_run_id: str = "",
    dry_run: bool = False,
) -> None:
    """CPU-gate the dataset, then dispatch the accepted 64E/Top-2 model to H100."""

    _require_source_commit(source_commit)
    request = _production.ProductionRequest(
        run_id=run_id,
        dataset_dir=dataset_dir,
        total_steps=steps,
        precision=precision,
        microbatch_size=microbatch_size,
        source_commit=source_commit,
        resume=resume or None,
        sequences_per_block=sequences_per_block or None,
        checkpoint_every_steps=checkpoint_every_steps,
        keep_last_checkpoints=keep_last_checkpoints,
        milestone_every_steps=milestone_every_steps,
        max_wall_seconds=max_wall_seconds,
        validation_blocks=None if validation_blocks < 0 else validation_blocks,
        compile_mode=compile_mode,
        allow_partial_corpus=allow_partial_corpus,
        dataset_shard_bucket=dataset_shard_bucket,
        dataset_shard_run_id=dataset_shard_run_id,
    )
    payload = _production.request_payload(request)
    payload["provider"] = "modal"
    payload["gpu"] = "H100"
    payload["command"] = _production.build_training_command(
        request, run_root=RUN_ROOT, schedule=_production.DISPLAY_SCHEDULE
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    if dry_run:
        return

    prepared = prepare_production_cpu.remote(_production.request_payload(request))
    if not isinstance(prepared, dict) or prepared.get("status") != "ready":
        raise RuntimeError(f"CPU production preparation did not authorize H100 dispatch: {prepared!r}")
    if prepared.get("identity") != _production.accepted_identity():
        raise RuntimeError("CPU production preparation returned a different model identity")
    result = train_production_h100.remote(_production.request_payload(request))
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if result.get("status") == "incomplete":
        raise SystemExit(f"segment ended at step {result.get('steps_reached')} before the target: corpus exhausted")


if __name__ == "__main__":
    raise SystemExit("use `modal run modal/moe_production_launch.py ...`")
