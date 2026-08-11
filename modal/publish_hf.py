#!/usr/bin/env python3
"""Publish a verified Modal training checkpoint to Hugging Face.

This is intentionally separate from the live trainer's legacy dataset-keyed
remote checkpoint protocol. Modal Volume remains the exact-resume transport;
this command exports the latest verified model/run artifact under a
model-specific Hugging Face namespace and can require final-run completion.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import modal

LOCAL_REPO = Path(__file__).resolve().parents[1]
LOCAL_MODAL = Path(__file__).resolve().parent
REMOTE_REPO = Path("/root/small-llm")
RUN_ROOT = Path("/runs")
APP_NAME = "small-llm-hf-publication"
_CHECKPOINT_ID = re.compile(r"^step-(\d{8})$")

sys.path.insert(0, str(LOCAL_MODAL))
from profiles import canonical_run_id, resolve_presets  # noqa: E402

RUN_VOLUME = modal.Volume.from_name("small-llm-runs", create_if_missing=False)
TRAINING_SECRET = modal.Secret.from_name("small-llm-training")

IMAGE = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install("huggingface-hub>=0.30,<2")
    .env({"PYTHONPATH": str(REMOTE_REPO), "PYTHONUNBUFFERED": "1"})
    .add_local_dir(
        LOCAL_REPO,
        remote_path=str(REMOTE_REPO),
        copy=False,
        ignore=[".git/**", ".venv/**", ".pytest_cache/**", "**/__pycache__/**", "*.pyc"],
    )
)
app = modal.App(APP_NAME, image=IMAGE)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError(f"expected JSON object: {path}")
    return dict(value)


def _latest_verified_checkpoint(checkpoint_dir: Path) -> tuple[Path, int]:
    from dataset.src.joint_checkpoint import verify_local_manifest

    if not checkpoint_dir.is_dir():
        raise RuntimeError(f"checkpoint directory does not exist: {checkpoint_dir}")
    valid: list[tuple[int, Path]] = []
    for root in checkpoint_dir.iterdir():
        match = _CHECKPOINT_ID.fullmatch(root.name) if root.is_dir() else None
        if match is None:
            continue
        try:
            verify_local_manifest(root)
            payload = _json(root / "checkpoint.json")
            pipeline = payload.get("pipeline_state")
            last = pipeline.get("last_consumed_block_id") if isinstance(pipeline, Mapping) else None
            step = int(match.group(1))
            if isinstance(last, int) and not isinstance(last, bool) and last == step - 1:
                valid.append((step, root))
        except Exception as error:  # noqa: BLE001 - invalid checkpoints must never publish
            print(f"Ignoring invalid checkpoint {root}: {type(error).__name__}: {error}", flush=True)
    if not valid:
        raise RuntimeError(f"no verified checkpoints found under {checkpoint_dir}")
    step, root = max(valid, key=lambda item: item[0])
    return root, step


@app.function(
    timeout=6 * 60 * 60,
    secrets=[TRAINING_SECRET],
    volumes={str(RUN_ROOT): RUN_VOLUME.with_mount_options(read_only=True)},
)
def publish_checkpoint(model: str, tokens: str, require_complete: bool = False) -> dict[str, object]:
    from huggingface_hub import HfApi

    model_preset, token_preset = resolve_presets(model, tokens)
    run_id = canonical_run_id(model_preset, token_preset)
    run_dir = RUN_ROOT / run_id
    checkpoint_dir = run_dir / "checkpoints"
    plan_path = run_dir / "qualification_plan.json"
    runtime_path = run_dir / "modal_runtime.json"

    if not plan_path.is_file():
        raise RuntimeError(f"missing qualification plan: {plan_path}")
    if not runtime_path.is_file():
        raise RuntimeError(f"missing Modal runtime contract: {runtime_path}")

    plan = _json(plan_path)
    trainer = plan.get("trainer")
    if not isinstance(trainer, Mapping):
        raise RuntimeError("qualification plan has no trainer section")
    total_steps = int(trainer["steps"])

    checkpoint, completed_steps = _latest_verified_checkpoint(checkpoint_dir)
    is_final = completed_steps == total_steps
    if require_complete and not is_final:
        raise RuntimeError(
            f"refusing to publish an incomplete run: latest verified step {completed_steps}, "
            f"planned final step {total_steps}"
        )

    repo_id = os.environ.get("SMALL_LLM_HF_REPO_ID")
    token = os.environ.get("HF_TOKEN")
    if not repo_id:
        raise RuntimeError("SMALL_LLM_HF_REPO_ID is not configured in small-llm-training secret")
    if not token:
        raise RuntimeError("HF_TOKEN is not configured in small-llm-training secret")

    runtime = _json(runtime_path)
    checkpoint_id = checkpoint.name
    prefix = f"models/{run_id}/{checkpoint_id}"
    metadata_path = f"models/{run_id}/artifact.json"
    metadata = {
        "version": 1,
        "artifact_type": (
            "small-llm-final-joint-checkpoint" if is_final else "small-llm-live-joint-checkpoint"
        ),
        "run_id": run_id,
        "model_label": model_preset.label,
        "model_parameters_nominal": model_preset.parameters,
        "trainer_model_size": model_preset.trainer_size,
        "token_label": token_preset.label,
        "training_tokens_nominal": token_preset.tokens,
        "dataset_profile": token_preset.dataset_profile,
        "checkpoint_id": checkpoint_id,
        "completed_steps": completed_steps,
        "total_steps": total_steps,
        "is_final": is_final,
        "source_commit": runtime.get("source_commit"),
        "dataset_run_id": runtime.get("dataset_run_id"),
        "precision": runtime.get("precision"),
        "microbatch_size": runtime.get("microbatch_size"),
        "huggingface_path": prefix,
        "verification": "dataset.src.joint_checkpoint.verify_local_manifest passed before upload",
    }

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=True,
        exist_ok=True,
    )
    folder_commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=checkpoint,
        path_in_repo=prefix,
        commit_message=f"Publish {run_id} checkpoint {checkpoint_id}",
    )
    metadata_commit = api.upload_file(
        repo_id=repo_id,
        repo_type="model",
        path_or_fileobj=(json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        path_in_repo=metadata_path,
        commit_message=f"Point {run_id} to checkpoint {checkpoint_id}",
    )
    return {
        "status": "published",
        "repo_id": repo_id,
        "run_id": run_id,
        "checkpoint_id": checkpoint_id,
        "completed_steps": completed_steps,
        "total_steps": total_steps,
        "is_final": is_final,
        "path_in_repo": prefix,
        "metadata_path": metadata_path,
        "checkpoint_commit": getattr(folder_commit, "oid", None),
        "metadata_commit": getattr(metadata_commit, "oid", None),
    }


@app.local_entrypoint()
def main(
    model: str = "100M",
    tokens: str = "2B",
    require_complete: bool = False,
) -> None:
    model_preset, token_preset = resolve_presets(model, tokens)
    print(
        json.dumps(
            {
                "action": "publish_verified_checkpoint_to_huggingface",
                "model": model_preset.label,
                "tokens": token_preset.label,
                "run_id": canonical_run_id(model_preset, token_preset),
                "source": "small-llm-runs Modal Volume",
                "require_complete": require_complete,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    result = publish_checkpoint.remote(
        model_preset.label,
        token_preset.label,
        require_complete,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    raise SystemExit(
        "Use: modal run modal/publish_hf.py --model 100M --tokens 2B "
        "[--require-complete]"
    )
