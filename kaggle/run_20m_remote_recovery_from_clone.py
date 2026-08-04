#!/usr/bin/env python3
"""Qualify private remote publication and empty-environment recovery on Kaggle.

Usage:
    %cd /kaggle/working/Small-LLM
    !git pull --ff-only
    !python kaggle/run_20m_remote_recovery_from_clone.py

This is a bounded recovery test, not another full repeatability run. It trains
through update 25 once, publishes the verified joint checkpoint, runs five local
reference updates, restores the checkpoint plus two Drive shards into a fresh
empty root, and runs the same five updates from the restored state. The local
and remote-restored trajectories and semantic checkpoints must match exactly.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
LOCAL_HELPER_PATH = Path(__file__).with_name("run_20m_local_resume_from_clone.py")
WORK = Path("/kaggle/working/small-llm-remote-recovery-controller")
LATEST = Path("/kaggle/working/small_llm_remote_recovery_summary.json")
COMMIT = "45d1da4a1ac3f18cf6ce02b8439672f10e2c8b4c"
BOUNDARY = 25
CONTINUATION_STEPS = 5
FINAL_STEP = BOUNDARY + CONTINUATION_STEPS
CHECKPOINT_25 = "step-00000025"
CHECKPOINT_30 = "step-00000030"


def load_local_helper():
    spec = importlib.util.spec_from_file_location(
        "small_llm_local_resume_helper", LOCAL_HELPER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load repository helper: {LOCAL_HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


local = load_local_helper()
helper = local.helper
base = local.base
GateFailure = base.GateFailure
helper.WORK = WORK


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_id_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Private remote checkpoint publication and empty-root recovery."
    )
    parser.add_argument(
        "--launch-commit",
        default=os.environ.get("SMALL_LLM_LAUNCH_COMMIT", COMMIT),
    )
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument(
        "--wandb-run-prefix",
        default=os.environ.get("SMALL_LLM_WANDB_REMOTE_PREFIX"),
    )
    return parser.parse_args(argv)


def secret_json(name: str, destination: Path) -> Path:
    value = base.secret(name)
    if not value:
        raise GateFailure(f"Missing required Kaggle Secret: {name}")
    try:
        payload = json.loads(value)
    except (TypeError, ValueError) as error:
        raise GateFailure(f"Kaggle Secret {name} is not valid JSON") from error
    if not isinstance(payload, Mapping) or not payload.get("refresh_token"):
        raise GateFailure(
            f"Kaggle Secret {name} must contain authorized-user OAuth JSON "
            "with a refresh_token"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
    )
    destination.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return destination


def set_flag(command: list[str], flag: str, value: str) -> None:
    local.set_flag(command, flag, value)


def make_trainer_command(
    uv: str,
    dataset: Path,
    manifest: Path,
    checkpoints: Path,
    *,
    run_id: str,
    run_name: str,
    entity: str | None,
    steps: int,
    resume: str | None = None,
) -> list[str]:
    command = local.trainer_command(
        uv,
        dataset,
        checkpoints,
        run_id=run_id,
        run_name=run_name,
        entity=entity,
        steps=steps,
        resume=resume,
    )
    set_flag(command, "--dataset-manifest", str(manifest))
    set_flag(command, "--evaluation-every-steps", "0")
    set_flag(command, "--validation-blocks", "0")
    if resume:
        set_flag(command, "--wandb-resume", "allow")
    tags_index = command.index("--wandb-tags")
    suffix = command[tags_index + 1 :]
    entity_suffix: list[str] = []
    if "--wandb-entity" in suffix:
        entity_index = suffix.index("--wandb-entity")
        entity_suffix = suffix[entity_index : entity_index + 2]
    command[tags_index + 1 :] = [
        "20m",
        "t4",
        "remote-recovery",
        "empty-environment",
    ] + entity_suffix
    return command


def make_publish_command(
    uv: str,
    dataset: Path,
    checkpoints: Path,
    *,
    run_id: str,
    run_name: str,
    entity: str | None,
    repo_id: str,
) -> list[str]:
    command = make_trainer_command(
        uv,
        dataset,
        dataset / "manifest.json",
        checkpoints,
        run_id=run_id,
        run_name=run_name,
        entity=entity,
        steps=BOUNDARY,
    )
    command.extend(
        [
            "--remote-publish-every-steps",
            str(BOUNDARY),
            "--remote-drive-manifest",
            str(dataset / "drive_manifest.json"),
            "--remote-checkpoint-repo",
            repo_id,
            "--remote-token-env",
            "HF_TOKEN",
            "--remote-create-repo",
        ]
    )
    return command


def parse_remote_publication(path: Path) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(item, Mapping) and isinstance(
            item.get("remote_publication"), Mapping
        ):
            events.append(dict(item["remote_publication"]))
    if not events:
        raise GateFailure("Publisher run emitted no remote_publication event")
    final = events[-1]
    if final.get("checkpoint_id") != CHECKPOINT_25:
        raise GateFailure(f"Unexpected published checkpoint: {final!r}")
    return {"events": events, "final": final}


RESTORE_SCRIPT = r'''
from __future__ import annotations
import json
import os
import shutil
import sys
from pathlib import Path

from googleapiclient.discovery import build

from dataset.drive_auth import load_authorized_user_credentials
from dataset.src.joint_checkpoint import restore_on_empty_vps
from dataset.src.remote import (
    GoogleDriveShardStore,
    HuggingFaceCheckpointStore,
    TwoPhaseCheckpointPublisher,
    sha256_path,
)

repo_id, run_id, token_path, destination, output = sys.argv[1:6]
destination = Path(destination)
output = Path(output)
if destination.exists():
    shutil.rmtree(destination)
destination.mkdir(parents=True)
if any(destination.iterdir()):
    raise RuntimeError("restore destination is not empty")

credentials = load_authorized_user_credentials(token_path)
service = build("drive", "v3", credentials=credentials, cache_discovery=False)
drive = GoogleDriveShardStore(service, "direct-file-id-download-only")
hub = HuggingFaceCheckpointStore(
    repo_id,
    token=os.environ["HF_TOKEN"],
    private=True,
)
publisher = TwoPhaseCheckpointPublisher(hub, run_id=run_id)
pointer_path = f"run/{run_id}/latest.json"
pointer = hub.read_json(pointer_path)
if pointer is None:
    raise RuntimeError(f"remote latest pointer is missing: {pointer_path}")
restored = restore_on_empty_vps(
    publisher=publisher,
    store=drive,
    run_id=run_id,
    destination=destination,
    checkpoint_pointer=pointer,
    prefetch_shards=2,
)

drive_manifest = json.loads((restored / "drive_manifest.json").read_text())
checkpoint = json.loads((restored / "checkpoint.json").read_text())
pipeline = checkpoint.get("pipeline_state", {})
if pipeline.get("last_consumed_block_id") != 24:
    raise RuntimeError("restored checkpoint does not point to next block 25")

expected = {}
for entry in drive_manifest["shards"]:
    if entry.get("filename") in {
        "train/train-000000.bin",
        "train/train-000001.bin",
    }:
        expected[entry["filename"]] = {
            "byte_size": int(entry["byte_size"]),
            "sha256": str(entry["local_sha256"]),
        }
if set(expected) != {
    "train/train-000000.bin",
    "train/train-000001.bin",
}:
    raise RuntimeError("Drive manifest does not contain the expected first two train shards")

actual = {}
for path in sorted((destination / "cache").rglob("*.bin")):
    relative = path.relative_to(destination / "cache").as_posix()
    actual[relative] = {
        "byte_size": path.stat().st_size,
        "sha256": sha256_path(path),
    }
if actual != expected:
    raise RuntimeError(
        "restored shard set or hashes differ from the Drive manifest: "
        f"expected={expected}, actual={actual}"
    )

result = {
    "pointer_path": pointer_path,
    "pointer": dict(pointer),
    "restored_checkpoint": str(restored),
    "last_consumed_block_id": 24,
    "next_block_id": 25,
    "prefetched_shards": actual,
    "prefetched_shard_count": len(actual),
    "destination_was_empty": True,
}
output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, sort_keys=True))
'''


def restore_remote_checkpoint(
    uv: str,
    worktree: Path,
    evidence: Path,
    env: Mapping[str, str],
    *,
    repo_id: str,
    run_id: str,
    token_path: Path,
    destination: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    script = evidence / "restore-empty-environment.py"
    output = evidence / "restore-empty-environment.json"
    script.write_text(RESTORE_SCRIPT, encoding="utf-8")
    stage = base.run(
        [
            uv,
            "run",
            "--python",
            "3.13",
            "--extra",
            "model",
            "--with-requirements",
            "dataset/requirements-remote.txt",
            "python",
            str(script),
            repo_id,
            run_id,
            str(token_path),
            str(destination),
            str(output),
        ],
        name="restore-empty-environment",
        evidence=evidence,
        cwd=worktree,
        env=env,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    if result.get("prefetched_shard_count") != 2:
        raise GateFailure("Remote restore did not prefetch exactly two train shards")
    if result.get("next_block_id") != BOUNDARY:
        raise GateFailure("Remote restore did not identify block 25 as the next block")
    return result, stage


def verify_single_checkpoint(
    uv: str,
    worktree: Path,
    evidence: Path,
    env: Mapping[str, str],
    root: Path,
    checkpoint_id: str,
    label: str,
) -> dict[str, Any]:
    code = (
        "import json,sys; from pathlib import Path; "
        "from dataset.src.joint_checkpoint import verify_local_manifest; "
        "root=Path(sys.argv[1]); "
        "result=verify_local_manifest(root); "
        "print(json.dumps({'root':str(root),'files':len(result['files'])},sort_keys=True))"
    )
    return base.run(
        [
            uv,
            "run",
            "--python",
            "3.13",
            "--extra",
            "model",
            "python",
            "-c",
            code,
            str(root / checkpoint_id),
        ],
        name=f"verify-checkpoint-{label}",
        evidence=evidence,
        cwd=worktree,
        env=env,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = arguments(argv)
    WORK.mkdir(parents=True, exist_ok=True)
    evidence = WORK / f"small-llm-remote-recovery-{stamp()}"
    evidence.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "started_utc": utc_now(),
        "status": "running",
        "authorization": "none",
        "scope": (
            "private checkpoint publication, two-shard Drive prefetch, and "
            "five-update exact empty-environment continuation"
        ),
        "training_budget": {
            "publisher_steps": BOUNDARY,
            "local_continuation_steps": CONTINUATION_STEPS,
            "remote_continuation_steps": CONTINUATION_STEPS,
            "total_executed_updates": BOUNDARY + 2 * CONTINUATION_STEPS,
            "reason": (
                "50-step stability, repeatability, and local resume already passed; "
                "this gate isolates remote durability and restoration"
            ),
        },
        "gradient_clipping_policy": {
            "accepted_for_qualification": True,
            "max_grad_norm": 1.0,
        },
        "evidence": str(evidence),
        "stages": [],
    }
    try:
        summary["environment"] = base.check_environment()
        wandb_key = base.secret("WANDB_API_KEY")
        entity = base.secret("WANDB_ENTITY", required=False)
        hf_token = base.secret("HF_TOKEN")
        hf_repo_id = base.secret("SMALL_LLM_HF_REPO_ID")
        if not wandb_key or not hf_token or not hf_repo_id:
            raise GateFailure(
                "WANDB_API_KEY, HF_TOKEN, and SMALL_LLM_HF_REPO_ID are required"
            )
        oauth_path = secret_json(
            "GOOGLE_DRIVE_OAUTH_TOKEN_JSON",
            evidence / ".secrets" / "google-drive-authorized-user.json",
        )

        worktree, checkout, checkout_stage = helper.prepare_worktree(
            args.launch_commit, evidence
        )
        summary["checkout"] = checkout
        summary["stages"].append(checkout_stage)
        env = {
            "WANDB_API_KEY": wandb_key,
            "HF_TOKEN": hf_token,
            "SMALL_LLM_HF_REPO_ID": hf_repo_id,
            "SMALL_LLM_GOOGLE_OAUTH_TOKEN": str(oauth_path),
            "UV_LINK_MODE": "copy",
            "PYTHONUNBUFFERED": "1",
            "HF_HOME": str(evidence / "hf-client-cache"),
        }

        summary["stages"].append(
            base.run(
                [sys.executable, "-m", "pip", "install", "-q", "uv"],
                name="install-uv",
                evidence=evidence,
            )
        )
        uv = base.find_uv()
        summary["stages"].append(
            base.run(
                [uv, "python", "install", "3.13"],
                name="install-python-3.13",
                evidence=evidence,
                cwd=worktree,
                env=env,
            )
        )
        summary["stages"].append(
            base.run(
                [uv, "run", "--python", "3.13", "python", "--version"],
                name="python-version",
                evidence=evidence,
                cwd=worktree,
                env=env,
            )
        )

        dataset, inspected = base.find_dataset(args.dataset_dir)
        summary["dataset"] = {"path": str(dataset), "inspected": inspected}
        summary["stages"].append(
            base.run(
                [
                    uv,
                    "run",
                    "--python",
                    "3.13",
                    "--with-requirements",
                    "dataset/requirements-remote.txt",
                    "python",
                    "-m",
                    "dataset.main",
                    "verify",
                    "--output-dir",
                    str(dataset),
                    "--full-scan",
                ],
                name="dataset-full-scan",
                evidence=evidence,
                cwd=worktree,
                env=env,
            )
        )
        plan_path = evidence / "qualification_plan.json"
        summary["stages"].append(
            base.run(
                [
                    uv,
                    "run",
                    "--python",
                    "3.13",
                    "python",
                    "-m",
                    "dataset.qualification_20m_report",
                    "--dataset-dir",
                    str(dataset),
                    "--drive-manifest",
                    str(dataset / "drive_manifest.json"),
                    "--output",
                    str(plan_path),
                ],
                name="qualification-plan",
                evidence=evidence,
                cwd=worktree,
                env=env,
            )
        )
        summary["plan"] = base.validate_plan(plan_path)
        remote_run_id = str(summary["plan"]["identity"]["drive_run_id"])

        prefix = args.wandb_run_prefix or f"20m-t4-remote-{run_id_stamp()}"
        publish_wandb_id = f"{prefix}-publisher"
        local_wandb_id = f"{prefix}-local-reference"
        remote_wandb_id = f"{prefix}-remote-restored"
        source_checkpoints = evidence / "publisher" / "checkpoints"

        publish_command = make_publish_command(
            uv,
            dataset,
            source_checkpoints,
            run_id=publish_wandb_id,
            run_name="20M T4 remote checkpoint publisher",
            entity=entity,
            repo_id=hf_repo_id,
        )
        summary["stages"].append(
            base.run(
                publish_command,
                name="trainer-publish-through-25",
                evidence=evidence,
                cwd=worktree,
                env=env,
            )
        )
        publisher_log = evidence / "trainer-publish-through-25.log"
        local.parse_training_log(
            publisher_log,
            expected_steps=range(1, BOUNDARY + 1),
            expected_validation_count=0,
            expected_final_checkpoint=CHECKPOINT_25,
        )
        publication = parse_remote_publication(publisher_log)
        summary["stages"].append(
            verify_single_checkpoint(
                uv,
                worktree,
                evidence,
                env,
                source_checkpoints,
                CHECKPOINT_25,
                "published-source-25",
            )
        )

        local_command = make_trainer_command(
            uv,
            dataset,
            dataset / "manifest.json",
            source_checkpoints,
            run_id=local_wandb_id,
            run_name="20M T4 local post-publication reference",
            entity=entity,
            steps=CONTINUATION_STEPS,
            resume=CHECKPOINT_25,
        )
        summary["stages"].append(
            base.run(
                local_command,
                name="trainer-local-reference-25-to-30",
                evidence=evidence,
                cwd=worktree,
                env=env,
            )
        )
        local_continuation = local.parse_training_log(
            evidence / "trainer-local-reference-25-to-30.log",
            expected_steps=range(BOUNDARY + 1, FINAL_STEP + 1),
            expected_validation_count=0,
            expected_final_checkpoint=CHECKPOINT_30,
        )

        empty_root = evidence / "empty-environment"
        if empty_root.exists():
            shutil.rmtree(empty_root)
        if (evidence / "hf-client-cache").exists():
            shutil.rmtree(evidence / "hf-client-cache")
        restore, restore_stage = restore_remote_checkpoint(
            uv,
            worktree,
            evidence,
            env,
            repo_id=hf_repo_id,
            run_id=remote_run_id,
            token_path=oauth_path,
            destination=empty_root,
        )
        summary["stages"].append(restore_stage)
        restored_checkpoints = empty_root / "checkpoints"
        restored_manifest = (
            restored_checkpoints / CHECKPOINT_25 / "drive_manifest.json"
        )
        summary["stages"].append(
            verify_single_checkpoint(
                uv,
                worktree,
                evidence,
                env,
                restored_checkpoints,
                CHECKPOINT_25,
                "remote-restored-25",
            )
        )

        semantic_25, semantic_25_stage = local.compare_checkpoint_semantics(
            uv,
            worktree,
            evidence,
            env,
            source_checkpoints / CHECKPOINT_25,
            restored_checkpoints / CHECKPOINT_25,
            "remote-step-25",
        )
        summary["stages"].append(semantic_25_stage)

        remote_command = make_trainer_command(
            uv,
            empty_root / "cache",
            restored_manifest,
            restored_checkpoints,
            run_id=remote_wandb_id,
            run_name="20M T4 empty-environment restored continuation",
            entity=entity,
            steps=CONTINUATION_STEPS,
            resume=CHECKPOINT_25,
        )
        summary["stages"].append(
            base.run(
                remote_command,
                name="trainer-remote-restored-25-to-30",
                evidence=evidence,
                cwd=worktree,
                env=env,
            )
        )
        remote_continuation = local.parse_training_log(
            evidence / "trainer-remote-restored-25-to-30.log",
            expected_steps=range(BOUNDARY + 1, FINAL_STEP + 1),
            expected_validation_count=0,
            expected_final_checkpoint=CHECKPOINT_30,
        )

        metric_comparison = helper.compare_rows(
            local_continuation["rows"], remote_continuation["rows"]
        )
        if (
            metric_comparison.get("numeric_trajectory_exact") is not True
            or metric_comparison.get("discrete_trajectory_exact") is not True
        ):
            raise GateFailure(
                "Remote-restored continuation differs from local continuation"
            )

        semantic_30, semantic_30_stage = local.compare_checkpoint_semantics(
            uv,
            worktree,
            evidence,
            env,
            source_checkpoints / CHECKPOINT_30,
            restored_checkpoints / CHECKPOINT_30,
            "remote-step-30",
        )
        summary["stages"].append(semantic_30_stage)

        summary["runs"] = {
            "publisher": {
                "wandb_run_id": publish_wandb_id,
                "checkpoint_dir": str(source_checkpoints),
                "steps": [1, BOUNDARY],
                "publication": publication,
            },
            "local_reference": {
                "wandb_run_id": local_wandb_id,
                "steps": [BOUNDARY + 1, FINAL_STEP],
                "final_checkpoint": str(source_checkpoints / CHECKPOINT_30),
            },
            "remote_restored": {
                "wandb_run_id": remote_wandb_id,
                "steps": [BOUNDARY + 1, FINAL_STEP],
                "checkpoint_dir": str(restored_checkpoints),
            },
        }
        summary["remote_restore"] = restore
        summary["comparison"] = {
            "metric_trajectory": metric_comparison,
            "checkpoint_semantics": {
                "step_25_source_vs_remote": semantic_25,
                "step_30_local_vs_remote": semantic_30,
            },
            "resume_class": "exact_remote_empty_environment_recovery",
        }
        summary["checkpoint_trees"] = {
            "source_step_25": helper.checkpoint_digest(
                source_checkpoints / CHECKPOINT_25
            ),
            "remote_step_25": helper.checkpoint_digest(
                restored_checkpoints / CHECKPOINT_25
            ),
            "local_step_30": helper.checkpoint_digest(
                source_checkpoints / CHECKPOINT_30
            ),
            "remote_step_30": helper.checkpoint_digest(
                restored_checkpoints / CHECKPOINT_30
            ),
        }
        summary["status"] = "passed_remote_empty_environment_recovery"
        summary["authorization"] = "full_306_run_ready_for_explicit_launch"
        summary["finished_utc"] = utc_now()
        base.write_json(evidence / "summary.json", summary)
        base.write_json(LATEST, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except BaseException as error:
        summary["status"] = "failed"
        summary["authorization"] = "none"
        summary["finished_utc"] = utc_now()
        summary["error"] = f"{type(error).__name__}: {error}"
        base.write_json(evidence / "summary.json", summary)
        base.write_json(LATEST, summary)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
