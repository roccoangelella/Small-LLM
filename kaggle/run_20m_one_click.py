#!/usr/bin/env python3
"""One-click Kaggle launcher for Small-LLM's 20M qualification preflight.

Kaggle requirements:
- NVIDIA T4 accelerator
- Internet enabled
- accepted private qualification dataset attached
- Kaggle Secrets: GITHUB_TOKEN and WANDB_API_KEY
- optional Kaggle Secret: WANDB_ENTITY

Run in one notebook cell:
    !python /kaggle/working/kaggle_20m_one_click.py

The launcher fails closed. It pins the launch commit, runs the offline suite,
runs the T4 harness, identifies and fully verifies the accepted dataset,
regenerates the exact plan, and runs the 20-update W&B preflight. It does not
start the complete 306-update segment; recovery and repeatability gates remain.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_URL = "https://github.com/roccoangelella/Small-LLM.git"
DEFAULT_COMMIT = "45d1da4a1ac3f18cf6ce02b8439672f10e2c8b4c"
MANIFEST_SHA256 = "1e5ee8f372b77b6728288610dbe7cce74d833be21e53d1538bc5a890229b18bb"
DRIVE_MANIFEST_SHA256 = "fbb29ee0d0102658e1274e39d6647cf56a6dcb685e0f566b1736847dcc4fbe84"
WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
LATEST = WORK / "small_llm_qualification_summary.json"

EXPECTED_PLAN: dict[str, Any] = {
    "version": 1,
    "qualification_profile": "20m-one-pass-v1",
    "accepted_source_tokens": 10_000_662,
    "train_source_tokens": 9_991_872,
    "validation_source_tokens": 8_790,
    "context_length": 2_048,
    "sequences_per_block": 16,
    "train": {"target_tokens": 10_006_528},
    "trainer": {
        "steps": 306,
        "passes": 1,
        "full_block_target_tokens": 32_768,
        "schedule": "wsd",
        "warmup_updates": 16,
        "stable_updates": 228,
        "decay_updates": 62,
        "warmup_tokens": 524_288,
        "stable_tokens": 7_471_104,
        "decay_tokens": 2_011_136,
        "minimum_lr_ratio": 0.1,
        "validation_blocks": 1,
    },
}


class GateFailure(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def safe_cmd(command: Sequence[str]) -> str:
    return shlex.join(
        "<redacted-authorization>" if "AUTHORIZATION:" in part.upper() else part
        for part in command
    )


def run(
    command: Sequence[str],
    *,
    name: str,
    evidence: Path,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    shown: Sequence[str] | None = None,
) -> dict[str, Any]:
    log_path = evidence / f"{name}.log"
    exit_path = evidence / f"{name}.exit-code"
    print(f"\n=== {name} ===\n$ {safe_cmd(shown or command)}", flush=True)
    started = time.perf_counter()
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command), cwd=str(cwd) if cwd else None, env=merged_env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = process.wait()
    exit_path.write_text(f"{code}\n", encoding="utf-8")
    result = {
        "name": name,
        "exit_code": code,
        "elapsed_seconds": time.perf_counter() - started,
        "log": str(log_path),
        "log_sha256": sha256(log_path),
    }
    if code:
        raise GateFailure(f"{name} failed with exit code {code}; see {log_path}")
    return result


def secret(name: str, required: bool = True) -> str | None:
    try:
        from kaggle_secrets import UserSecretsClient
    except Exception as error:
        raise GateFailure("Run this file inside Kaggle; kaggle_secrets is unavailable") from error
    try:
        value = UserSecretsClient().get_secret(name)
    except Exception:
        value = None
    if required and not value:
        raise GateFailure(f"Missing required Kaggle Secret: {name}")
    return value


def check_environment() -> dict[str, Any]:
    if not WORK.is_dir() or not INPUT.is_dir():
        raise GateFailure("Expected /kaggle/working and /kaggle/input")
    if shutil.which("nvidia-smi") is None:
        raise GateFailure("Enable a GPU accelerator")
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    if result.returncode or not rows:
        raise GateFailure(f"nvidia-smi failed: {result.stdout.strip()}")
    if "T4" not in rows[0].upper():
        raise GateFailure(f"Qualification requires NVIDIA T4; detected {rows[0]!r}")
    return {"gpu": rows[0]}


def find_uv() -> str:
    for candidate in [shutil.which("uv"), str(Path.home() / ".local/bin/uv"), "/opt/conda/bin/uv"]:
        if candidate and Path(candidate).is_file():
            return candidate
    raise GateFailure("uv was installed but its executable was not found")


def clone_repo(token: str, commit: str, repo: Path, evidence: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if repo.exists():
        shutil.rmtree(repo)
    auth = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    clone = ["git", "-c", f"http.extraHeader=AUTHORIZATION: basic {auth}", "clone", "--no-checkout", REPO_URL, str(repo)]
    fetch = ["git", "-c", f"http.extraHeader=AUTHORIZATION: basic {auth}", "fetch", "origin", commit]
    stages = [
        run(clone, name="git-clone", evidence=evidence, shown=["git", "clone", "--no-checkout", REPO_URL, str(repo)]),
        run(fetch, name="git-fetch", evidence=evidence, cwd=repo, shown=["git", "fetch", "origin", commit]),
        run(["git", "checkout", "--detach", commit], name="git-checkout", evidence=evidence, cwd=repo),
    ]
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()
    if actual != commit or dirty:
        raise GateFailure(f"Exact checkout failed: actual={actual}, dirty={bool(dirty)}")
    return {"requested": commit, "actual": actual, "clean": True, "detached": True}, stages


def find_dataset(explicit: Path | None) -> tuple[Path, list[dict[str, Any]]]:
    roots = [explicit.resolve()] if explicit else sorted({p.parent for p in INPUT.rglob("manifest.json")})
    inspected: list[dict[str, Any]] = []
    matches: list[Path] = []
    for root in roots:
        manifest = root / "manifest.json"
        drive = root / "drive_manifest.json"
        row: dict[str, Any] = {"root": str(root)}
        if manifest.is_file():
            row["manifest_sha256"] = sha256(manifest)
        if drive.is_file():
            row["drive_manifest_sha256"] = sha256(drive)
        row["train"] = (root / "train").is_dir()
        row["validation"] = (root / "validation").is_dir()
        inspected.append(row)
        if (
            row.get("manifest_sha256") == MANIFEST_SHA256
            and row.get("drive_manifest_sha256") == DRIVE_MANIFEST_SHA256
            and row["train"] and row["validation"]
        ):
            matches.append(root)
    if not matches:
        raise GateFailure("Accepted dataset not found. Inspected:\n" + json.dumps(inspected, indent=2))
    return sorted(matches, key=lambda p: (len(p.parts), str(p)))[0], inspected


def nested_check(actual: Mapping[str, Any], expected: Mapping[str, Any], prefix: str = "") -> None:
    for key, wanted in expected.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in actual:
            raise GateFailure(f"Plan missing {path}")
        got = actual[key]
        if isinstance(wanted, Mapping):
            if not isinstance(got, Mapping):
                raise GateFailure(f"Plan field {path} is not an object")
            nested_check(got, wanted, path)
        elif isinstance(wanted, float):
            if not isinstance(got, (int, float)) or not math.isclose(float(got), wanted, abs_tol=1e-12, rel_tol=0):
                raise GateFailure(f"Plan mismatch at {path}: expected {wanted!r}, got {got!r}")
        elif got != wanted:
            raise GateFailure(f"Plan mismatch at {path}: expected {wanted!r}, got {got!r}")


def validate_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(plan, Mapping):
        raise GateFailure("Generated plan is not an object")
    nested_check(plan, EXPECTED_PLAN)
    identity = plan.get("identity")
    if not isinstance(identity, Mapping):
        raise GateFailure("Generated plan has no identity")
    if identity.get("manifest_sha256") != MANIFEST_SHA256:
        raise GateFailure("Generated plan has the wrong manifest hash")
    if identity.get("drive_manifest_sha256") != DRIVE_MANIFEST_SHA256:
        raise GateFailure("Generated plan has the wrong Drive-manifest hash")
    return dict(plan)


def validate_t4(path: Path, commit: str) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    env = report.get("environment", {})
    parity = report.get("parity")
    benches = report.get("benchmarks")
    if report.get("schema_version") != 2 or env.get("git_commit") != commit:
        raise GateFailure("T4 report identity mismatch")
    if "T4" not in str(env.get("device_name", "")).upper():
        raise GateFailure("T4 report was not produced on a T4")
    if not isinstance(parity, list) or len(parity) != 12 or any(not isinstance(p, Mapping) or p.get("status") != "pass" for p in parity):
        raise GateFailure("Not all 12 T4 parity cases passed")
    if not isinstance(benches, list):
        raise GateFailure("T4 benchmark list is missing")
    fixed = [b for b in benches if isinstance(b, Mapping) and b.get("architecture") == "gdn2_hybrid" and b.get("chunk_size") == 32 and b.get("precision") == "fp16"]
    if len(fixed) != 1 or fixed[0].get("status") != "pass" or fixed[0].get("overflow_count") != 0:
        raise GateFailure("GDN-2 chunk-32 FP16 did not pass cleanly")
    recommendation = report.get("recommendation", {})
    if recommendation.get("status") == "blocked":
        raise GateFailure("T4 harness found no viable candidate")
    return {"environment": env, "chunk32_fp16": fixed[0], "recommendation": recommendation}


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch-commit", default=os.environ.get("SMALL_LLM_LAUNCH_COMMIT", DEFAULT_COMMIT))
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--gates-only", action="store_true")
    parser.add_argument("--wandb-run-id", default=os.environ.get("SMALL_LLM_WANDB_RUN_ID", "20m-t4-preflight-001"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = arguments(argv)
    evidence = WORK / f"small-llm-qualification-{stamp()}"
    evidence.mkdir(parents=True, exist_ok=False)
    repo = WORK / "Small-LLM"
    t4_json = evidence / "t4_qualification.json"
    plan_json = evidence / "qualification_plan.json"
    checkpoints = evidence / "checkpoints-preflight"
    summary: dict[str, Any] = {"schema_version": 1, "started_utc": now(), "status": "running", "authorization": "none", "evidence": str(evidence), "stages": []}
    try:
        summary["environment"] = check_environment()
        github_token = secret("GITHUB_TOKEN")
        wandb_key = secret("WANDB_API_KEY", required=not args.gates_only)
        wandb_entity = secret("WANDB_ENTITY", required=False)
        assert github_token is not None
        summary["checkout"], checkout_stages = clone_repo(github_token, args.launch_commit, repo, evidence)
        summary["stages"].extend(checkout_stages)

        summary["stages"].append(run([sys.executable, "-m", "pip", "install", "-q", "uv"], name="install-uv", evidence=evidence))
        uv = find_uv()
        summary["stages"].append(run([uv, "python", "install", "3.13"], name="install-python-3.13", evidence=evidence, cwd=repo))
        summary["stages"].append(run([uv, "run", "--python", "3.13", "python", "--version"], name="python-version", evidence=evidence, cwd=repo))

        common = [uv, "run", "--python", "3.13"]
        offline = common + ["--extra", "model", "--with", "wandb==0.26.1", "--with-requirements", "dataset/requirements-remote.txt", "python", "-m", "unittest", "discover", "-v"]
        summary["stages"].append(run(offline, name="offline-tests", evidence=evidence, cwd=repo))

        t4 = common + ["--extra", "model", "python", "-m", "tests.t4_qualification", "--require-t4", "--chunk-sizes", "16", "32", "64", "--precisions", "fp32", "fp16", "--sequence-length", "2048", "--batch-size", "1", "--warmup-steps", "1", "--measure-steps", "3", "--include-plan-b", "--output", str(t4_json)]
        summary["stages"].append(run(t4, name="t4-qualification", evidence=evidence, cwd=repo))
        summary["t4"] = validate_t4(t4_json, args.launch_commit)

        dataset, inspected = find_dataset(args.dataset_dir)
        summary["dataset"] = {"selected": str(dataset), "inspected": inspected}
        verify = common + ["--with-requirements", "dataset/requirements-remote.txt", "python", "-m", "dataset.main", "verify", "--output-dir", str(dataset), "--full-scan"]
        summary["stages"].append(run(verify, name="dataset-full-scan", evidence=evidence, cwd=repo))
        report = common + ["python", "-m", "dataset.qualification", "report", "--profile", "20m-10m", "--dataset-dir", str(dataset), "--drive-manifest", str(dataset / "drive_manifest.json"), "--output", str(plan_json)]
        summary["stages"].append(run(report, name="qualification-plan", evidence=evidence, cwd=repo))
        plan = validate_plan(plan_json)
        summary["plan"] = plan
        summary["authorization"] = "20_update_preflight"

        if args.gates_only:
            summary["status"] = "passed_gates_only"
            summary["next_action"] = "Rerun without --gates-only for the 20-update preflight"
        else:
            assert wandb_key is not None
            trainer = plan["trainer"]
            cmd = common + [
                "--extra", "model", "--with", "wandb==0.26.1", "--with-requirements", "dataset/requirements-remote.txt",
                "python", "-m", "trainer", "--dataset-dir", str(dataset), "--dataset-manifest", str(dataset / "manifest.json"),
                "--checkpoint-dir", str(checkpoints), "--steps", "20", "--sequences-per-block", "16", "--model-size", "smoke",
                "--architecture", "gdn2_hybrid", "--gdn-chunk-size", "32", "--initialization", "normal", "--optimizer", "hybrid_muon_adamw",
                "--device", "cuda", "--precision", "fp16", "--microbatch-size", "1", "--learning-rate", "3e-4", "--weight-decay", "0.1",
                "--muon-momentum", "0.95", "--muon-lr-multiplier", "1.0", "--muon-update-rms", "0.18", "--muon-weight-decay", "0.1",
                "--max-grad-norm", "1.0", "--schedule", "constant", "--evaluation-every-steps", "20", "--validation-blocks", str(trainer["validation_blocks"]),
                "--seed", "17", "--wandb-mode", "online", "--wandb-project", "Small-LLM", "--wandb-run-id", args.wandb_run_id,
                "--wandb-run-name", "20M T4 preflight", "--wandb-resume", "never",
            ]
            if wandb_entity:
                cmd += ["--wandb-entity", wandb_entity]
            cmd += ["--wandb-tags", "20m", "t4", "preflight"]
            env = {"WANDB_API_KEY": wandb_key}
            if wandb_entity:
                env["WANDB_ENTITY"] = wandb_entity
            summary["stages"].append(run(cmd, name="trainer-preflight-20", evidence=evidence, cwd=repo, env=env))
            final_checkpoint = checkpoints / "step-00000020" / "checkpoint.json"
            if not final_checkpoint.is_file():
                raise GateFailure(f"Final checkpoint missing: {final_checkpoint}")
            summary["preflight"] = {"steps": 20, "checkpoint": str(final_checkpoint), "wandb_run_id": args.wandb_run_id}
            summary["status"] = "passed_preflight"
            summary["authorization"] = "post_preflight_review_only"
            summary["next_action"] = "Review telemetry, freeze thresholds, then run A/A, interruption/resume, and remote recovery gates before the 306-update segment"

        summary["finished_utc"] = now()
        write_json(evidence / "summary.json", summary)
        write_json(LATEST, summary)
        print("\n=== RESULT ===")
        print(json.dumps(summary, indent=2, sort_keys=True))
        print(f"\nLatest summary: {LATEST}\nEvidence: {evidence}")
        return 0
    except BaseException as error:
        summary.update({"status": "failed", "authorization": "none", "finished_utc": now(), "failure": {"type": type(error).__name__, "message": str(error)}})
        try:
            write_json(evidence / "summary.json", summary)
            write_json(LATEST, summary)
        finally:
            print(json.dumps(summary, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
