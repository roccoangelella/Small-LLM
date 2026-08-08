#!/usr/bin/env python3
"""Restore the verified 500M checkpoint remotely, then run decay telemetry.

This wrapper is intentionally diagnostic-only. It restores the latest verified
checkpoint from the existing Hugging Face two-phase checkpoint namespace,
verifies that it is step 4000 and matches the attached 500M Drive manifest,
converts the opaque trainer_state.pkl mapping into a temporary torch checkpoint
for the existing telemetry probe, and then runs that forward-only probe.

It does not start the trainer, initialize W&B, run backward, step an optimizer,
advance the scheduler/data cursor, or publish a checkpoint.

Kaggle:
    python kaggle/run_gdn2_step4000_decay_telemetry_remote.py
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
KAGGLE_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(KAGGLE_DIR) not in sys.path:
    sys.path.insert(0, str(KAGGLE_DIR))

RUN_ID = "20m-500m-dataset-001"
EXPECTED_CHECKPOINT_ID = "step-00004000"
WORK = Path("/kaggle/working/gdn2-step4000-decay-telemetry")
# restore_on_empty_vps() creates its own `checkpoints/` and `cache/` children
# below the destination, so pass a neutral restore root rather than a path that
# is already named `checkpoints`.
RESTORE_DESTINATION = WORK / "restore"
CHECKPOINTS_ROOT = RESTORE_DESTINATION / "checkpoints"
RESTORE_RESULT = WORK / "restore.json"
TEMP_TORCH_CHECKPOINT = WORK / "step-00004000-trainer-state.pt"

RESTORE_SCRIPT = r'''
import json, os, sys
from pathlib import Path
from dataset.src.joint_checkpoint import restore_on_empty_vps
from dataset.src.remote import HuggingFaceCheckpointStore, TwoPhaseCheckpointPublisher, sha256_path

repo_id, run_id, destination, attached_manifest, output = sys.argv[1:6]
destination, attached_manifest, output = Path(destination), Path(attached_manifest), Path(output)
store = HuggingFaceCheckpointStore(repo_id, token=os.environ["HF_TOKEN"], private=True)
pointer_path = f"run/{run_id}/latest.json"
pointer = store.read_json(pointer_path)
if pointer is None:
    output.write_text(json.dumps({"status": "missing", "pointer_path": pointer_path}) + "\n")
    raise SystemExit(0)
root = restore_on_empty_vps(
    publisher=TwoPhaseCheckpointPublisher(store, run_id=run_id),
    store=None,
    run_id=run_id,
    destination=destination,
    checkpoint_pointer=pointer,
    prefetch_shards=0,
)
if sha256_path(root / "drive_manifest.json") != sha256_path(attached_manifest):
    raise RuntimeError("remote checkpoint Drive manifest differs from attached dataset")
checkpoint = json.loads((root / "checkpoint.json").read_text())
last = checkpoint.get("pipeline_state", {}).get("last_consumed_block_id")
if not isinstance(last, int):
    raise RuntimeError("restored checkpoint has no integer block cursor")
result = {
    "status": "restored",
    "checkpoint_id": pointer["checkpoint_id"],
    "checkpoint_root": str(root),
    "last_consumed_block_id": last,
}
output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, sort_keys=True))
'''


def secret(name: str) -> str:
    value = os.environ.get(name)
    if value:
        return value
    try:
        from kaggle_secrets import UserSecretsClient
        value = UserSecretsClient().get_secret(name)
    except Exception as error:
        raise SystemExit(f"Missing Kaggle secret {name}: {error}") from error
    if not value:
        raise SystemExit(f"Missing Kaggle secret {name}")
    return value


def ensure_uv() -> str:
    found = shutil.which("uv")
    if found:
        return found
    print("[setup] installing uv", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "uv"], check=True)
    found = shutil.which("uv")
    if not found:
        raise SystemExit("uv installation completed but executable was not found")
    return found


def restore(dataset_root: Path) -> Path:
    hf_token = secret("HF_TOKEN")
    hf_repo = secret("SMALL_LLM_HF_REPO_ID")
    uv = ensure_uv()

    WORK.mkdir(parents=True, exist_ok=True)
    if RESTORE_DESTINATION.exists():
        shutil.rmtree(RESTORE_DESTINATION)
    RESTORE_DESTINATION.mkdir(parents=True, exist_ok=True)
    restore_script = WORK / "restore_verified_checkpoint.py"
    restore_script.write_text(RESTORE_SCRIPT, encoding="utf-8")
    if RESTORE_RESULT.exists():
        RESTORE_RESULT.unlink()

    env = os.environ.copy()
    env.update({
        "HF_TOKEN": hf_token,
        "SMALL_LLM_HF_REPO_ID": hf_repo,
        "PYTHONUNBUFFERED": "1",
        "UV_LINK_MODE": "copy",
    })
    command = [
        uv,
        "run",
        "--python",
        "3.13",
        "--with-requirements",
        str(ROOT / "dataset" / "requirements-remote.txt"),
        "python",
        str(restore_script),
        hf_repo,
        RUN_ID,
        str(RESTORE_DESTINATION),
        str(dataset_root / "drive_manifest.json"),
        str(RESTORE_RESULT),
    ]
    print("[checkpoint] restoring latest verified remote 500M checkpoint", flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    if not RESTORE_RESULT.is_file():
        raise SystemExit("remote restore produced no result file")
    result = json.loads(RESTORE_RESULT.read_text(encoding="utf-8"))
    if result.get("status") != "restored":
        raise SystemExit(f"remote checkpoint restore did not succeed: {result}")
    if result.get("checkpoint_id") != EXPECTED_CHECKPOINT_ID:
        raise SystemExit(
            f"expected latest verified checkpoint {EXPECTED_CHECKPOINT_ID}, "
            f"got {result.get('checkpoint_id')!r}; refusing to inspect a different trajectory point"
        )
    if result.get("last_consumed_block_id") != 3999:
        raise SystemExit(
            "restored step-4000 checkpoint does not report last_consumed_block_id=3999"
        )
    root = Path(str(result.get("checkpoint_root", ""))).resolve()
    expected = (CHECKPOINTS_ROOT / EXPECTED_CHECKPOINT_ID).resolve()
    if root != expected or not root.is_dir():
        raise SystemExit(f"restored checkpoint root mismatch: {root} != {expected}")
    print(f"[checkpoint] verified {EXPECTED_CHECKPOINT_ID} at {root}", flush=True)
    return root


def make_torch_checkpoint(checkpoint_root: Path) -> Path:
    trainer_state = checkpoint_root / "trainer_state.pkl"
    if not trainer_state.is_file():
        raise SystemExit(f"restored checkpoint has no trainer_state.pkl: {checkpoint_root}")
    with trainer_state.open("rb") as handle:
        raw: Any = pickle.load(handle)
    if not isinstance(raw, dict):
        raise SystemExit("trainer_state.pkl did not contain a state dictionary")
    if raw.get("global_step") != 4000:
        raise SystemExit(f"trainer state global_step={raw.get('global_step')!r}, expected 4000")
    if not isinstance(raw.get("model"), dict) or not isinstance(raw.get("model_config"), dict):
        raise SystemExit("trainer state is missing model/model_config")

    import torch
    torch.save(raw, TEMP_TORCH_CHECKPOINT)
    print(f"[checkpoint] materialized telemetry-only torch state: {TEMP_TORCH_CHECKPOINT}", flush=True)
    return TEMP_TORCH_CHECKPOINT


def main() -> int:
    import run_gdn2_step4000_decay_telemetry as telemetry

    # Reuse the telemetry probe's exact dataset-selection rule so restoration is
    # cryptographically checked against the same attached Drive manifest that
    # supplies block 4000.
    dataset_root, _ = telemetry.discover_dataset(None, telemetry.TARGET_BLOCK)
    checkpoint_root = restore(dataset_root)
    torch_checkpoint = make_torch_checkpoint(checkpoint_root)

    original = sys.argv[:]
    try:
        sys.argv = [
            str(Path(telemetry.__file__).resolve()),
            "--checkpoint",
            str(torch_checkpoint),
            "--dataset-dir",
            str(dataset_root),
        ]
        return int(telemetry.main())
    finally:
        sys.argv = original


if __name__ == "__main__":
    raise SystemExit(main())
