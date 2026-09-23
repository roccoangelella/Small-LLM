#!/usr/bin/env python3
"""Modal H100 launcher for the accepted production MoE identity."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
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
CONTROL = _modal.Dict.from_name("small-llm-production-control", create_if_missing=True)


def _control(request):
    import os
    from moe_remote_control import RemoteControl
    from moe_checkpoint_transport import bucket_id
    from dataset.src.hf_bucket_checkpoint import HuggingFaceBucketCheckpointStore
    from dataset.src.hf_bucket_shards import HuggingFaceBucketShardStore
    return RemoteControl(request, CONTROL,
        HuggingFaceBucketCheckpointStore(bucket_id(request), token=os.environ.get("HF_TOKEN")),
        HuggingFaceBucketShardStore(request.dataset_shard_bucket, token=os.environ.get("HF_TOKEN")))


def _dataset_volume(request: _production.ProductionRequest):
    destination = Path(request.dataset_dir).resolve()
    for root, volume in ((DATA_ROOT, DATA_VOLUME), (CACHE_ROOT, CACHE_VOLUME), (RUN_ROOT, RUN_VOLUME)):
        if destination.is_relative_to(root.resolve()):
            return volume
    raise ValueError("streaming dataset_dir must be inside a mounted data, cache or run volume")


def _runtime_request(payload: dict[str, object]) -> _production.ProductionRequest:
    request = _production.request_from_payload(payload)
    # Modal mounts are aliases; pass their canonical paths to the existing
    # transport, which correctly rejects symlinks inside dataset/checkpoint trees.
    _dataset_volume(request)
    return replace(request, dataset_dir=str(Path(request.dataset_dir).resolve()))


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
    retries=0,  # CPU claim/staging failures require reconciliation, not automatic retry.
    secrets=[TRAINING_SECRET],
    volumes={
        str(DATA_ROOT): DATA_VOLUME,
        str(RUN_ROOT): RUN_VOLUME,
        str(CACHE_ROOT): CACHE_VOLUME,
    },
)
def prepare_production_cpu(payload: dict[str, object], segment_id: str = "") -> dict[str, object]:
    """Decode a real training block with the accepted 8,000-token semantic bound."""

    request = _runtime_request(payload)
    if segment_id:
        control = _control(request)
        pointer = control.checkpoints.read_json(f"run/{request.run_id}/latest.json")
        if request.resume != "latest" or not pointer:
            raise RuntimeError("Durable continuation requires existing HF latest")
        checkpoint = pointer.get("checkpoint_id", "")
        import re
        if not re.fullmatch(r"step-\d{8}", checkpoint):
            raise RuntimeError("Invalid HF latest checkpoint")
        control.check_launch(int(checkpoint[5:]))
        control.prepare_claim(segment_id)
    try:
        if request.checkpoint_bucket:
            RUN_VOLUME.reload()
        if request.streaming:
            volume = _dataset_volume(request)
            volume.reload()
        result = _production.prepare_dataset(request, run_root=RUN_ROOT.resolve())
        if request.streaming:
            volume.commit()
        if request.checkpoint_bucket:
            RUN_VOLUME.commit()
        if segment_id and result.get("training_complete"):
            control.finish(segment_id, {"status": "complete", "training_complete": True})
        return result
    except Exception as error:
        if segment_id:
            control.status(segment_id, "prepare_failed", error_type=type(error).__name__)
        raise



@app.function(
    gpu="H100",
    timeout=24 * 60 * 60,
    retries=0,  # Recovery requires verified latest/single-writer preflight, never blind GPU retries.
    single_use_containers=True,
    secrets=[TRAINING_SECRET],
    volumes={
        str(DATA_ROOT): DATA_VOLUME,
        str(RUN_ROOT): RUN_VOLUME,
        str(CACHE_ROOT): CACHE_VOLUME,
    },
)
def train_production_h100(payload: dict[str, object], segment_id: str = "") -> dict[str, object]:
    """Run only ``MoEModelConfig.accepted()`` on the production H100 path."""

    request = _runtime_request(payload)
    if not segment_id:
        if request.checkpoint_bucket:
            RUN_VOLUME.reload()
        if request.streaming:
            _dataset_volume(request).reload()
        return _production.run_provider_payload(
            _production.request_payload(request), run_root=RUN_ROOT.resolve(),
            repo_root=REMOTE_REPO, volume_commit=RUN_VOLUME.commit)
    from moe_remote_control import CorpusHold, ClaimBlocked
    control = _control(request)
    try:
        control.claim_training(segment_id, _modal.current_function_call_id())
    except ClaimBlocked as error:
        # Do not overwrite the first attempt's status on a platform reschedule.
        return {"status": "blocked_retry", "error_type": type(error).__name__, "segment_id": segment_id}
    try:
        RUN_VOLUME.reload()
        _dataset_volume(request).reload()
        plan = _production.resolve_resume(request, run_root=RUN_ROOT.resolve())
        control.last_step = int(plan["completed_steps"])
        control.check_launch(control.last_step)
        result = _production.run_provider_payload(
            _production.request_payload(request), run_root=RUN_ROOT.resolve(),
            repo_root=REMOTE_REPO, volume_commit=RUN_VOLUME.commit,
            event_callback=lambda line: control.observe(line, segment_id))
    except CorpusHold:
        result = {"status": "data_hold", "segment_id": segment_id,
                  "last_observed_step": control.last_step}
    except Exception as error:
        import traceback
        traceback.print_exc()
        result = {"status": "failed", "segment_id": segment_id,
                  "error_type": type(error).__name__, "last_observed_step": control.last_step}
    control.finish(segment_id, result)
    print(json.dumps({"durable_result": result}), flush=True)
    return result


@app.local_entrypoint()
def main(
    run_id: str,
    dataset_dir: str,
    steps: int,
    source_commit: str,
    precision: str = "bf16",
    microbatch_size: int = 64,
    validation_microbatch_size: int = 1,
    async_checkpoint_upload: bool = False,
    resume_source_commit: str = "",
    resume: str = "latest",
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
    checkpoint_bucket: str = "auto",
    wandb_entity: str = "",
    wandb_project: str = "Small-LLM",
    dry_run: bool = False,
    durable: bool = False,
    segment_id: str = "",
) -> None:
    """CPU-gate the dataset, then dispatch the accepted 64E/Top-2 model to H100."""

    _require_source_commit(source_commit)
    request = _production.ProductionRequest(
        run_id=run_id,
        dataset_dir=dataset_dir,
        total_steps=steps,
        precision=precision,
        microbatch_size=microbatch_size,
        validation_microbatch_size=validation_microbatch_size,
        async_checkpoint_upload=async_checkpoint_upload,
        resume_source_commit=resume_source_commit,
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
        checkpoint_bucket=checkpoint_bucket,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
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

    if durable:
        from moe_remote_control import validate_segment
        validate_segment(segment_id)
        prepared = prepare_production_cpu.remote(_production.request_payload(request), segment_id)
    else:
        prepared = prepare_production_cpu.remote(_production.request_payload(request))
    if not isinstance(prepared, dict) or prepared.get("status") != "ready":
        raise RuntimeError(f"CPU production preparation did not authorize H100 dispatch: {prepared!r}")
    if prepared.get("identity") != _production.accepted_identity():
        raise RuntimeError("CPU production preparation returned a different model identity")
    if prepared.get("training_complete"):
        print(json.dumps(prepared, sort_keys=True), flush=True)
        return
    if durable:
        call = train_production_h100.spawn(_production.request_payload(request), segment_id)
        record = {"run_id": run_id, "segment_id": segment_id, "call_id": call.object_id}
        CONTROL.put(("dispatch", run_id, segment_id), record)
        print(json.dumps({"durable_dispatch": record}), flush=True)
        return
    result = train_production_h100.remote(_production.request_payload(request))
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if not isinstance(result, dict):
        raise RuntimeError("Modal returned no training result; recover with --resume latest")
    if result.get("status") == "incomplete":
        raise SystemExit(f"segment ended at step {result.get('steps_reached')} before the target: corpus exhausted")


if __name__ == "__main__":
    raise SystemExit("use `modal run modal/moe_production_launch.py ...`")
