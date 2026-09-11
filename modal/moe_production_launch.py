#!/usr/bin/env python3
"""Modal H100 launcher for the accepted production MoE identity."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
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


def _require_source_commit(source_commit: str) -> None:
    actual = _base._local_source_commit()
    if source_commit != actual:
        raise RuntimeError(
            f"--source-commit {source_commit} does not match clean checkout HEAD {actual}"
        )


@app.function(
    cpu=2,
    memory=8192,
    timeout=30 * 60,
    retries=1,
    secrets=[TRAINING_SECRET],
    volumes={
        str(DATA_ROOT): DATA_VOLUME.with_mount_options(read_only=True),
        str(RUN_ROOT): RUN_VOLUME,
        str(CACHE_ROOT): CACHE_VOLUME,
    },
)
def prepare_production_cpu(payload: dict[str, object]) -> dict[str, object]:
    """Decode a real training block with the accepted 8,000-token semantic bound."""

    request = _production.request_from_payload(payload)
    return _production.prepare_dataset(request)


@app.function(
    gpu="H100",
    timeout=24 * 60 * 60,
    retries=3,
    single_use_containers=True,
    secrets=[TRAINING_SECRET],
    volumes={
        str(DATA_ROOT): DATA_VOLUME.with_mount_options(read_only=True),
        str(RUN_ROOT): RUN_VOLUME,
        str(CACHE_ROOT): CACHE_VOLUME,
    },
)
def train_production_h100(payload: dict[str, object]) -> dict[str, object]:
    """Run only ``MoEModelConfig.accepted()`` on the production H100 path."""

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
    precision: str = "fp16",
    microbatch_size: int = 8,
    resume: str = "",
    sequences_per_block: int = 0,
    checkpoint_every_steps: int = 0,
    validation_blocks: int = 0,
    dry_run: bool = False,
) -> None:
    """CPU-gate the dataset, then dispatch the accepted 64E/Top-2 model to H100."""

    _require_source_commit(source_commit)
    request = _production.ProductionRequest(
        run_id=run_id,
        dataset_dir=dataset_dir,
        steps=steps,
        precision=precision,
        microbatch_size=microbatch_size,
        source_commit=source_commit,
        resume=resume or None,
        sequences_per_block=sequences_per_block or None,
        checkpoint_every_steps=checkpoint_every_steps,
        validation_blocks=validation_blocks,
    )
    payload = _production.request_payload(request)
    payload["provider"] = "modal"
    payload["gpu"] = "H100"
    payload["command"] = _production.build_training_command(
        request, run_root=RUN_ROOT
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


if __name__ == "__main__":
    raise SystemExit("use `modal run modal/moe_production_launch.py ...`")
