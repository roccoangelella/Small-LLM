#!/usr/bin/env python3
"""Fail-closed Kaggle launcher for the authorized frozen 20M one-pass run."""

from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import run_20m_one_click as common

REPO = Path(__file__).resolve().parents[1]
COMMIT = common.DEFAULT_COMMIT
ROOT = common.WORK / "small-llm-20m-full-training"
WORKTREE = ROOT / "launch-worktree"
EVIDENCE = ROOT / ("evidence-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
CHECKPOINTS = common.WORK / "checkpoints-one-pass"
SUMMARY = common.WORK / "small_llm_20m_full_training_summary.json"
WANDB_RUN_ID = "20m-one-pass-001"


def repo_head() -> str:
    try:
        top = Path(subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=REPO, text=True
        ).strip()).resolve()
    except (OSError, subprocess.CalledProcessError) as error:
        raise common.GateFailure("Run from the cloned Small-LLM repository") from error
    if top != REPO.resolve():
        raise common.GateFailure(f"Repository root mismatch: {top}")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO, text=True,
    ).strip()
    if dirty:
        raise common.GateFailure("The controlling clone has tracked modifications")
    if subprocess.run(
        ["git", "cat-file", "-e", f"{COMMIT}^{{commit}}"], cwd=REPO,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode:
        raise common.GateFailure(f"Frozen launch commit {COMMIT} is missing")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def prepare_worktree() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    if WORKTREE.exists():
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(WORKTREE)], cwd=REPO,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if WORKTREE.exists():
            shutil.rmtree(WORKTREE)
    subprocess.run(["git", "worktree", "prune"], cwd=REPO, check=False)
    common.run(
        ["git", "worktree", "add", "--detach", str(WORKTREE), COMMIT],
        name="git-launch-worktree", evidence=EVIDENCE, cwd=REPO,
    )
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=WORKTREE, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=WORKTREE, text=True).strip()
    if actual != COMMIT or dirty:
        raise common.GateFailure(f"Frozen worktree mismatch: {actual}, dirty={bool(dirty)}")


def command(uv: str, dataset: Path, plan: dict, entity: str | None) -> list[str]:
    p = plan["trainer"]
    cmd = [
        uv, "run", "--python", "3.13", "--extra", "model",
        "--with", "wandb==0.26.1",
        "--with-requirements", "dataset/requirements-remote.txt",
        "python", "-m", "trainer",
        "--dataset-dir", str(dataset),
        "--dataset-manifest", str(dataset / "manifest.json"),
        "--checkpoint-dir", str(CHECKPOINTS),
        "--steps", str(p["steps"]),
        "--sequences-per-block", "16",
        "--model-size", "smoke",
        "--architecture", "gdn2_hybrid",
        "--gdn-chunk-size", "32",
        "--initialization", "normal",
        "--optimizer", "hybrid_muon_adamw",
        "--device", "cuda", "--precision", "fp16", "--microbatch-size", "1",
        "--learning-rate", "3e-4", "--weight-decay", "0.1",
        "--muon-momentum", "0.95", "--muon-lr-multiplier", "1.0",
        "--muon-update-rms", "0.18", "--muon-weight-decay", "0.1",
        "--max-grad-norm", "1.0", "--schedule", "wsd",
        "--warmup-tokens", str(p["warmup_tokens"]),
        "--stable-tokens", str(p["stable_tokens"]),
        "--decay-tokens", str(p["decay_tokens"]),
        "--minimum-lr-ratio", "0.1",
        "--checkpoint-every-steps", "25",
        "--evaluation-every-steps", "50",
        "--validation-blocks", str(p["validation_blocks"]),
        "--remote-publish-every-steps", "50",
        "--remote-drive-manifest", str(dataset / "drive_manifest.json"),
        "--remote-token-env", "HF_TOKEN", "--seed", "17",
        "--wandb-mode", "online", "--wandb-project", "Small-LLM",
        "--wandb-run-id", WANDB_RUN_ID,
        "--wandb-run-name", "20M one-pass qualification",
    ]
    if entity:
        cmd += ["--wandb-entity", entity]
    return cmd + ["--wandb-tags", "20m", "t4", "qualification", "one-pass"]


def main() -> int:
    state = {
        "started_utc": common.now(), "status": "initializing",
        "authorization": "full_306_run_authorized", "launch_commit": COMMIT,
        "wandb_run_id": WANDB_RUN_ID, "checkpoint_dir": str(CHECKPOINTS),
        "evidence_dir": str(EVIDENCE),
    }
    try:
        if CHECKPOINTS.exists():
            raise common.GateFailure(f"{CHECKPOINTS} exists; use explicit resume instead")
        environment = common.check_environment()
        controller_head = repo_head()
        wandb_key = common.secret("WANDB_API_KEY")
        hf_token = common.secret("HF_TOKEN")
        hf_repo = common.secret("SMALL_LLM_HF_REPO_ID")
        entity = common.secret("WANDB_ENTITY", required=False)
        assert wandb_key and hf_token and hf_repo

        EVIDENCE.mkdir(parents=True, exist_ok=False)
        state.update(status="preparing", environment=environment,
                     controller_head=controller_head, remote_checkpoint_repo=hf_repo)
        common.write_json(SUMMARY, state)
        prepare_worktree()
        dataset, inspected = common.find_dataset(None)
        state.update(dataset_dir=str(dataset), datasets_inspected=inspected)

        common.run([sys.executable, "-m", "pip", "install", "-q", "uv"],
                   name="install-uv", evidence=EVIDENCE, cwd=WORKTREE)
        uv = common.find_uv()
        common.run([uv, "python", "install", "3.13"],
                   name="install-python-3.13", evidence=EVIDENCE, cwd=WORKTREE)
        common.run([uv, "run", "--python", "3.13", "python", "--version"],
                   name="python-version", evidence=EVIDENCE, cwd=WORKTREE)
        base = [uv, "run", "--python", "3.13"]
        common.run(base + [
            "--with-requirements", "dataset/requirements-remote.txt",
            "python", "-m", "dataset.main", "verify",
            "--output-dir", str(dataset), "--full-scan",
        ], name="dataset-full-scan", evidence=EVIDENCE, cwd=WORKTREE)
        plan_path = EVIDENCE / "qualification_plan.json"
        common.run(base + [
            "python", "-m", "dataset.qualification_20m_report",
            "--dataset-dir", str(dataset),
            "--drive-manifest", str(dataset / "drive_manifest.json"),
            "--output", str(plan_path),
        ], name="qualification-plan", evidence=EVIDENCE, cwd=WORKTREE)
        plan = common.validate_plan(plan_path)

        cmd = command(uv, dataset, plan, entity)
        state.update(status="running", trainer_started_utc=common.now(),
                     plan=plan, trainer_command=cmd)
        common.write_json(SUMMARY, state)
        print(f"\nSTARTING AUTHORIZED 306-UPDATE RUN\ndataset: {dataset}\nW&B: {WANDB_RUN_ID}\n")
        common.run(
            cmd, name="trainer-306-updates", evidence=EVIDENCE, cwd=WORKTREE,
            env={"WANDB_API_KEY": wandb_key, "HF_TOKEN": hf_token,
                 "SMALL_LLM_HF_REPO_ID": hf_repo},
        )
        state.update(status="completed", completed_utc=common.now())
        common.write_json(SUMMARY, state)
        print(f"Full run completed. Summary: {SUMMARY}")
        return 0
    except KeyboardInterrupt:
        state.update(status="interrupted", finished_utc=common.now())
        common.write_json(SUMMARY, state)
        return 130
    except Exception as error:
        state.update(status="failed", finished_utc=common.now(),
                     error=f"{type(error).__name__}: {error}")
        common.write_json(SUMMARY, state)
        print(f"LAUNCH FAILED CLOSED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
