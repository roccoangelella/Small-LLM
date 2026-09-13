"""Cross-account MoE continuation using the existing verified bucket transport.

Dataset shards stay in their own bucket. The durability receipt binds the run
to the immutable dataset metadata, source commit and W&B identity.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

from dataset.src.checkpoint_sequence import complete_checkpoint, find_latest_complete_checkpoint
from dataset.src.hf_bucket_checkpoint import HuggingFaceBucketCheckpointStore
from dataset.src.joint_checkpoint import restore_on_empty_vps, _verify_published_checkpoint_manifest
from dataset.src.remote import TwoPhaseCheckpointPublisher, sha256_path
from dataset.src.storage import read_json, write_json_atomic
from trainer.remote_publication import promote_best_pointer

RECEIPT_FILENAME = "checkpoint_transport.json"


def bucket_id(request: object) -> str:
    if request.checkpoint_bucket != "auto":
        return request.checkpoint_bucket
    explicit = os.environ.get("SMALL_LLM_HF_CHECKPOINT_BUCKET_ID", "").strip()
    if explicit:
        return explicit
    repo = os.environ.get("SMALL_LLM_HF_REPO_ID", "").strip()
    if not repo:
        raise ValueError("set --checkpoint-bucket or SMALL_LLM_HF_REPO_ID for durable MoE checkpoints")
    return f"{repo}-checkpoints"


def receipt_path(request: object, run_root: Path) -> Path:
    return run_root / request.run_id / RECEIPT_FILENAME


def restore_checkpoint(request: object, *, run_root: Path) -> Mapping[str, object] | None:
    """Fetch a newer remote checkpoint before the CPU dataset gate, never fresh-start on a miss."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN is required for cross-provider checkpoint transport")
    store = HuggingFaceBucketCheckpointStore(
        bucket_id(request), token=token, private=True, create_bucket=request.resume == "new",
    )
    local_root = run_root / request.run_id / "checkpoints"
    local = find_latest_complete_checkpoint(local_root)
    pointer = store.read_json(f"run/{request.run_id}/latest.json")
    if request.resume == "new":
        if local is not None or pointer is not None:
            raise ValueError("run already has a checkpoint; use --resume latest or a new run ID")
        return None
    if pointer is not None:
        checkpoint_id = pointer.get("checkpoint_id")
        from dataset.src.checkpoint_sequence import checkpoint_step
        if not isinstance(checkpoint_id, str) or checkpoint_step(Path(checkpoint_id)) is None:
            raise ValueError("remote latest has an invalid checkpoint ID")
        remote_step = checkpoint_step(Path(checkpoint_id))
        if request.resume not in (None, "", "latest", checkpoint_id):
            raise ValueError("cannot rewind a shared run behind remote latest; use a new run ID")
        if local is None or remote_step > local["step"]:
            restored = restore_on_empty_vps(
                publisher=TwoPhaseCheckpointPublisher(store, run_id=request.run_id),
                store=None, run_id=request.run_id, destination=run_root / request.run_id,
                checkpoint_pointer=pointer, prefetch_shards=0,
            )
            local = complete_checkpoint(restored)
        elif remote_step == local["step"]:
            # Same step must identify the same snapshot, not a divergent writer.
            expected = {item["name"]: item["sha256"] for item in pointer["checkpoint_manifest"]["files"]}
            for name in ("trainer_state.pkl", "checkpoint.json", "local_manifest.json"):
                if sha256_path(local["path"] / name) != expected.get(name):
                    raise ValueError("local checkpoint diverges from published state at the same step")
            try:
                _verify_published_checkpoint_manifest(local["path"], pointer.get("checkpoint_manifest"))
            except (RuntimeError, OSError, ValueError):
                # Modal can commit the local checkpoint before publication adds
                # these two sidecars. Recover verified metadata without replacing
                # or adapting any saved trainer tensor.
                with tempfile.TemporaryDirectory(dir=run_root / request.run_id) as temporary:
                    verified = restore_on_empty_vps(
                        publisher=TwoPhaseCheckpointPublisher(store, run_id=request.run_id),
                        store=None, run_id=request.run_id, destination=Path(temporary),
                        checkpoint_pointer=pointer, prefetch_shards=0,
                    )
                    for name in ("drive_manifest.json", "checkpoint_manifest.json"):
                        staged = local["path"] / f".{name}.repair"
                        shutil.copyfile(verified / name, staged)
                        os.replace(staged, local["path"] / name)
                _verify_published_checkpoint_manifest(local["path"], pointer.get("checkpoint_manifest"))
    if local is None:
        raise ValueError("no verified checkpoint found; --resume new is required to start a new run")
    if request.resume not in (None, "", "latest", str(local["checkpoint_id"])):
        raise ValueError("explicit resume is not the latest recoverable checkpoint")
    saved_path = local["path"] / "drive_manifest.json"
    # A locally saved but not yet published snapshot has the run's CPU receipt.
    if not saved_path.is_file():
        saved_path = receipt_path(request, run_root)
    receipt = read_json(saved_path)
    if receipt.get("run_id") != request.run_id or receipt.get("version") != 1:
        raise ValueError("checkpoint transport receipt belongs to a different run")
    return receipt


def prepare_receipt(request: object, *, run_root: Path,
                    previous: Mapping[str, object] | None) -> None:
    dataset_root = Path(request.dataset_dir)
    entity = request.wandb_entity or os.environ.get("WANDB_ENTITY", "")
    if not entity:
        raise ValueError("set --wandb-entity so every provider uses the same W&B account")
    if not os.environ.get("WANDB_API_KEY"):
        raise ValueError("WANDB_API_KEY is required for production telemetry")
    from moe_production import resolve_schedule
    validation_blocks = (request.validation_blocks if request.validation_blocks is not None
                         else resolve_schedule(dataset_root)["validation_blocks"])
    if validation_blocks <= 0:
        raise ValueError("latest/best checkpoint retention requires held-out validation")
    receipt = {
        "version": 1, "run_id": request.run_id, "shards": [],
        "checkpoint_bucket": bucket_id(request),
        "source_commit": getattr(request, "resume_source_commit", "") or request.source_commit,
        "dataset_manifest_sha256": sha256_path(dataset_root / "manifest.json"),
        "run_contract_sha256": sha256_path(dataset_root / "run_contract.json"),
        "validation_blocks": validation_blocks,
        "wandb": {"entity": entity, "project": request.wandb_project, "id": request.run_id},
    }
    if previous is not None and dict(previous) != receipt:
        changed = sorted(key for key in set(previous) | set(receipt) if previous.get(key) != receipt.get(key))
        raise ValueError(f"cross-provider run identity mismatch: {changed}")
    if getattr(request, "resume_source_commit", ""):
        if previous is None or request.resume != "latest":
            raise ValueError("source transition cannot initialize a new run")
        checkpoint = find_latest_complete_checkpoint(run_root / request.run_id / "checkpoints")
        if checkpoint is None:
            raise ValueError("source transition requires a verified complete resume checkpoint")
        # v1 remains byte/schema compatible with the original executor for rollback.
        # The actual clean checkout commit is recorded separately in checkpoint.json.
    write_json_atomic(receipt_path(request, run_root), receipt)


def finalize_completed_run(request: object, *, run_root: Path) -> None:
    """Finish an interrupted final upload/best-pointer update on CPU, without renting a GPU."""
    checkpoint = find_latest_complete_checkpoint(run_root / request.run_id / "checkpoints")
    if checkpoint is None or checkpoint["step"] != request.total_steps:
        raise ValueError("cannot finalize an incomplete run")
    receipt = read_json(receipt_path(request, run_root))
    store = HuggingFaceBucketCheckpointStore(bucket_id(request), token=os.environ.get("HF_TOKEN"))
    pointer = store.read_json(f"run/{request.run_id}/latest.json")
    if pointer is None or pointer.get("checkpoint_id") != checkpoint["checkpoint_id"]:
        pointer = TwoPhaseCheckpointPublisher(store, run_id=request.run_id).publish(
            checkpoint["path"], checkpoint_id=checkpoint["checkpoint_id"], drive_manifest=receipt,
        )["latest"]
    loss = read_json(checkpoint["path"] / "checkpoint.json").get("validation_metrics", {}).get("loss")
    best = store.read_json(f"run/{request.run_id}/best.json")
    promote_best_pointer(store, run_id=request.run_id, latest=pointer,
                         metric=None if loss is None else -float(loss),
                         best_metric=None if best is None else best["metric"])
    store.prune_run_checkpoints(run_id=request.run_id, checkpoint_id=checkpoint["checkpoint_id"], keep_best=True)


def training_flags(request: object, *, run_root: Path, display: bool = False,
                   resumed: bool = False) -> list[str]:
    receipt = ({"checkpoint_bucket": request.checkpoint_bucket,
                "wandb": {"entity": request.wandb_entity or "WANDB_ENTITY", "project": request.wandb_project}}
               if display else read_json(receipt_path(request, run_root)))
    return [
        "--remote-checkpoint-bucket", receipt["checkpoint_bucket"],
        "--remote-drive-manifest", str(receipt_path(request, run_root)),
        "--remote-publish-every-steps", str(request.checkpoint_every_steps),
        "--remote-keep-latest-and-best",
        "--wandb-mode", "online", "--wandb-entity", receipt["wandb"]["entity"],
        "--wandb-project", receipt["wandb"]["project"],
        "--wandb-run-id", request.run_id, "--wandb-run-name", request.run_id,
        "--wandb-resume", "must" if resumed else "allow",
    ]
