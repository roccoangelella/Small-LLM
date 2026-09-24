#!/usr/bin/env python3
"""Read-only preflight or explicitly start run 003 from its verified HF latest.

This is a *single-device* MoE continuation on cuda:0. The second T4 is not
used: the canonical MoE engine has no distributed training implementation.
Never start while another provider is writing this W&B/HF run.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moe_checkpoint_transport import bucket_id
from moe_production import ProductionRequest, accepted_identity, prepare_dataset, run_provider_payload, request_payload

BRANCH = "moe-8e-top1"
RUN_ID = "moe-100b-superbpe-003"
RUN_ORIGIN = "5f08941028cce913f336a8aa2f8ce5fb6231ca4c"
DATASET_BUCKET = "abcastor/small-llm-corpus-100b-v2-dataset"
DATASET_RUN_ID = "moe-100b-superbpe-b64-dataset-003"
CHECKPOINT_BUCKET = "roccoangelella/small-llm-100m-qualification-checkpoints"
WANDB_ENTITY = "rocchissimo936-none"
WANDB_PROJECT = "Small-LLM"
TARGET_STEPS = 762940


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def clean_branch_commit() -> str:
    if _git("branch", "--show-current") != BRANCH:
        raise RuntimeError(f"expected the public {BRANCH} branch, not this checkout")
    if _git("status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("checkout is dirty; do not train with uncommitted code")
    commit = _git("rev-parse", "HEAD")
    remote = subprocess.check_output(
        ["git", "ls-remote", "https://github.com/roccoangelella/Small-LLM.git", f"refs/heads/{BRANCH}"],
        text=True, timeout=30,
    ).strip().split()
    if not remote or remote[0] != commit:
        raise RuntimeError("local checkout is not the public moe-8e-top1 HEAD")
    return commit


def remote_latest(request: ProductionRequest) -> tuple[str, dict[str, object]]:
    from dataset.src.hf_bucket_checkpoint import HuggingFaceBucketCheckpointStore
    from dataset.src.checkpoint_sequence import checkpoint_step

    store = HuggingFaceBucketCheckpointStore(bucket_id(request), token=os.environ["HF_TOKEN"], private=True)
    pointer = store.read_json(f"run/{RUN_ID}/latest.json")
    if pointer is None or not isinstance(pointer.get("checkpoint_id"), str):
        raise RuntimeError("no verified latest checkpoint; refusing a fresh run")
    checkpoint_id = str(pointer["checkpoint_id"])
    if checkpoint_step(Path(checkpoint_id)) is None:
        raise RuntimeError("invalid remote latest checkpoint ID")
    prefix = pointer.get("last_prefix")
    if not isinstance(prefix, str) or not prefix.startswith(f"run/{RUN_ID}/checkpoints/{checkpoint_id}/"):
        raise RuntimeError("remote latest points outside the expected run checkpoint")
    receipt = store.read_json(prefix + "/drive_manifest.json")
    expected = {"version": 1, "run_id": RUN_ID, "source_commit": RUN_ORIGIN,
                "checkpoint_bucket": CHECKPOINT_BUCKET,
                "wandb": {"entity": WANDB_ENTITY, "project": WANDB_PROJECT, "id": RUN_ID}}
    if receipt is None or any(receipt.get(key) != value for key, value in expected.items()):
        raise RuntimeError("latest checkpoint receipt does not match the pinned live run")
    return checkpoint_id, dict(receipt)


def wandb_state() -> str:
    import wandb

    run = wandb.Api(api_key=os.environ["WANDB_API_KEY"], timeout=30).run(
        f"{WANDB_ENTITY}/{WANDB_PROJECT}/{RUN_ID}")
    return str(run.state)


def kaggle_popen(command: list[str], **kwargs: object) -> subprocess.Popen:
    if command[:3] != [sys.executable, "-m", "MOE_model"]:
        raise RuntimeError("unexpected production training command; refusing to replace its entrypoint")
    return subprocess.Popen([sys.executable, str(ROOT / "kaggle" / "moe_train_single_t4.py"),
                             *command[3:]], **kwargs)


def request(commit: str, work_dir: Path) -> ProductionRequest:
    return ProductionRequest(
        run_id=RUN_ID, dataset_dir=str(work_dir / "dataset"), total_steps=TARGET_STEPS,
        precision="bf16", microbatch_size=1, validation_microbatch_size=1,
        source_commit=commit, resume_source_commit=RUN_ORIGIN, resume="latest",
        checkpoint_every_steps=1000, validation_blocks=16, keep_last_checkpoints=2,
        max_wall_seconds=9 * 60 * 60, compile_mode="off",  # T4: no qualified compile lane
        dataset_shard_bucket=DATASET_BUCKET, dataset_shard_run_id=DATASET_RUN_ID,
        checkpoint_bucket=CHECKPOINT_BUCKET, wandb_entity=WANDB_ENTITY,
        wandb_project=WANDB_PROJECT,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("/kaggle/working/moe-resume-003"))
    parser.add_argument("--start", action="store_true", help="Explicitly start one training segment (default: preflight only)")
    args = parser.parse_args(argv)
    for name in ("HF_TOKEN", "WANDB_API_KEY", "SMALL_LLM_HF_REPO_ID"):
        if not os.environ.get(name):
            raise RuntimeError(f"{name} is required")
    if os.environ["SMALL_LLM_HF_REPO_ID"].strip() + "-checkpoints" != CHECKPOINT_BUCKET:
        raise RuntimeError("HF repo ID does not select the live run's checkpoint bucket")
    commit = clean_branch_commit()
    req = request(commit, args.work_dir.resolve())
    checkpoint_id, receipt = remote_latest(req)
    state = wandb_state()
    print(json.dumps({"checkout": commit, "branch": BRANCH, "run_id": RUN_ID,
                      "checkpoint_id": checkpoint_id, "wandb_state": state,
                      "receipt_manifest_sha256": receipt["dataset_manifest_sha256"],
                      "model": accepted_identity(), "mode": "start" if args.start else "preflight"},
                     sort_keys=True), flush=True)
    if args.start and state == "running":
        raise RuntimeError("W&B run is still running; stop the previous writer before Kaggle starts")
    if args.start and state not in {"finished", "failed", "crashed", "killed"}:
        raise RuntimeError(f"W&B state {state!r} does not authorize a new writer")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    run_root = args.work_dir / "runs"
    prepared = prepare_dataset(req, run_root=run_root)
    if prepared.get("status") != "ready" or prepared.get("identity") != accepted_identity():
        raise RuntimeError("CPU gate did not verify the accepted MoE dataset")
    if remote_latest(req)[0] != checkpoint_id:
        raise RuntimeError("remote latest advanced during the CPU gate; retry from the new pointer")
    if args.start and wandb_state() == "running":
        raise RuntimeError("another W&B writer started during CPU preparation")
    print(json.dumps({"cpu_gate": prepared, "checkpoint_id": checkpoint_id}, sort_keys=True), flush=True)
    if not args.start:
        return 0
    result = run_provider_payload(
        request_payload(req), run_root=run_root, repo_root=ROOT, popen_factory=kaggle_popen,
    )
    print(json.dumps({"training_result": result}, sort_keys=True), flush=True)
    if result["status"] not in {"drained", "complete"}:
        raise RuntimeError(f"Kaggle segment did not finish safely: {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
