#!/usr/bin/env python3
"""Run the Small-LLM 50-update reference/A-A test from a Kaggle clone.

Usage:
    %cd /kaggle/working/Small-LLM
    !git pull --ff-only
    !python kaggle/run_20m_repeatability_from_clone.py

The controller clone stays on ``main``. Evidence is produced in a clean detached
worktree at the frozen launch commit. The accepted dataset is fully verified and
the exact 306-update plan is regenerated before two independent 50-update WSD
prefixes run with identical seed, data order, model, optimizer, and T4. Metrics
and step-25/step-50 checkpoints are compared. This measures repeatability; it
does not start or authorize the complete 306-update run.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = Path(__file__).with_name("run_20m_one_click.py")
WORK = Path("/kaggle/working/small-llm-repeatability-controller")
LATEST = Path("/kaggle/working/small_llm_repeatability_summary.json")
COMMIT = "45d1da4a1ac3f18cf6ce02b8439672f10e2c8b4c"
STEPS = 50
TOKENS_PER_STEP = 32_768


def load_base():
    spec = importlib.util.spec_from_file_location("small_llm_kaggle_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base()
GateFailure = base.GateFailure


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_id_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def prepare_worktree(commit: str, evidence: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    try:
        top = Path(subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=ROOT, text=True,
            stderr=subprocess.STDOUT,
        ).strip()).resolve()
    except (OSError, subprocess.CalledProcessError) as error:
        raise GateFailure("Run this file from a cloned Small-LLM repository") from error
    if top != ROOT.resolve():
        raise GateFailure(f"Repository root mismatch: expected {ROOT}, got {top}")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
    ).strip():
        raise GateFailure("The controlling clone has tracked modifications")
    controller_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    ).returncode:
        raise GateFailure(f"Frozen commit {commit} is absent; run git fetch origin")

    worktree = WORK / "Small-LLM"
    if worktree.exists():
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree)], cwd=ROOT,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )
        if worktree.exists():
            shutil.rmtree(worktree)
    subprocess.run(
        ["git", "worktree", "prune"], cwd=ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    )
    stage = base.run(
        ["git", "worktree", "add", "--detach", str(worktree), commit],
        name="git-launch-worktree", evidence=evidence, cwd=ROOT,
    )
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=worktree, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=worktree, text=True).strip()
    if actual != commit or dirty:
        raise GateFailure(f"Detached worktree mismatch: actual={actual}, dirty={bool(dirty)}")
    return worktree, {
        "requested": commit, "actual": actual, "clean": True, "detached": True,
        "controller_repo": str(ROOT), "controller_head": controller_head,
        "launch_worktree": str(worktree),
    }, stage


def require_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise GateFailure(f"Non-finite metric at {path}: {value!r}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            require_finite(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            require_finite(child, f"{path}[{index}]")


def parse_log(path: Path) -> dict[str, Any]:
    rows, validations, checkpoints = [], [], []
    final_id = None
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(item, Mapping):
            continue
        if {"step", "block_id", "loss", "gradient_norm"}.issubset(item):
            row = dict(item); require_finite(row); rows.append(row)
        elif isinstance(item.get("validation"), Mapping):
            validation = dict(item["validation"]); require_finite(validation); validations.append(validation)
        elif isinstance(item.get("local_checkpoint"), Mapping):
            checkpoints.append(dict(item["local_checkpoint"]))
        elif isinstance(item.get("checkpoint_id"), str):
            final_id = item["checkpoint_id"]
    if len(rows) != STEPS:
        raise GateFailure(f"Expected {STEPS} successful updates in {path}, found {len(rows)}")
    for step, row in enumerate(rows, 1):
        expected = {
            "step": step, "block_id": step - 1, "sequences": 16,
            "target_tokens": TOKENS_PER_STEP, "consumed_tokens": step * TOKENS_PER_STEP,
        }
        for key, wanted in expected.items():
            if row.get(key) != wanted:
                raise GateFailure(f"Trajectory mismatch at step {step}, {key}: {row.get(key)!r} != {wanted!r}")
    if len(validations) != 1 or final_id != "step-00000050":
        raise GateFailure(
            f"Expected one validation and final checkpoint step-00000050; "
            f"got validations={len(validations)}, checkpoint={final_id!r}"
        )
    return {"rows": rows, "validation": validations[0], "checkpoint_events": checkpoints}


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def summarize(parsed: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(parsed["rows"]); steady = rows[3:]
    loss = [float(row["loss"]) for row in rows]
    grad = [float(row["gradient_norm"]) for row in rows]
    speed = [float(row["tokens_per_second"]) for row in rows]
    steady_speed = [float(row["tokens_per_second"]) for row in steady]
    clipped = [bool(row.get("gradient_clipped", False)) for row in rows]
    scaler = [float(row.get("grad_scaler_scale", 1.0)) for row in rows]
    return {
        "steps": STEPS,
        "target_tokens": int(rows[-1]["consumed_tokens"]),
        "loss": {"first": loss[0], "last": loss[-1], "minimum": min(loss), "maximum": max(loss)},
        "learning_rate": {
            "first": float(rows[0]["learning_rate"]),
            "step_16": float(rows[15]["learning_rate"]),
            "step_17": float(rows[16]["learning_rate"]),
            "last": float(rows[-1]["learning_rate"]),
        },
        "gradient_norm": {
            "minimum": min(grad), "maximum": max(grad), "last": grad[-1],
            "first_10_median": statistics.median(grad[:10]),
            "last_10_median": statistics.median(grad[-10:]),
            "post_startup_median": statistics.median(grad[3:]),
            "post_startup_p95": percentile(grad[3:], 0.95),
        },
        "clipping": {"count": sum(clipped), "fraction": sum(clipped) / STEPS},
        "fp16": {
            "scaler_minimum": min(scaler), "scaler_maximum": max(scaler),
            "overflow_retries_total": sum(int(row.get("overflow_retries", 0)) for row in rows),
            "overflow_events_final": int(rows[-1].get("overflow_events_total", 0)),
        },
        "throughput": {
            "mean": statistics.fmean(speed), "minimum": min(speed), "maximum": max(speed),
            "post_startup_median": statistics.median(steady_speed),
            "post_startup_mad": statistics.median(
                abs(value - statistics.median(steady_speed)) for value in steady_speed
            ),
            "post_startup_p05": percentile(steady_speed, 0.05),
            "post_startup_p95": percentile(steady_speed, 0.95),
        },
        "memory": {
            "maximum_allocated_bytes": max(int(row["peak_memory_bytes"]) for row in rows),
            "maximum_reserved_bytes": max(int(row.get("peak_reserved_memory_bytes", 0)) for row in rows),
        },
        "data_wait_seconds_mean": statistics.fmean(
            float(row.get("data_wait_seconds", 0.0)) for row in rows
        ),
        "validation": dict(parsed["validation"]),
        "checkpoint_events": list(parsed["checkpoint_events"]),
    }


def flatten(value: Any, prefix: str, output: dict[str, Any]) -> None:
    if isinstance(value, Mapping):
        for key in sorted(value):
            flatten(value[key], f"{prefix}.{key}" if prefix else str(key), output)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            flatten(child, f"{prefix}[{index}]", output)
    elif isinstance(value, (bool, int, float, str)):
        output[prefix] = value


def compare_rows(reference: Sequence[Mapping[str, Any]], repeat: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    runtime = {"elapsed_seconds", "tokens_per_second", "data_wait_seconds", "peak_memory_bytes", "peak_reserved_memory_bytes"}
    discrete = {"step", "block_id", "sequences", "target_tokens", "consumed_tokens", "gradient_clipped", "overflow_retries", "overflow_events_total"}
    differences, compared = [], 0
    for step, (left, right) in enumerate(zip(reference, repeat, strict=True), 1):
        a, b = {}, {}
        flatten({key: value for key, value in left.items() if key not in runtime}, "", a)
        flatten({key: value for key, value in right.items() if key not in runtime}, "", b)
        if set(a) != set(b):
            raise GateFailure(f"Metric structure mismatch at step {step}")
        for path in sorted(a):
            if path in discrete or isinstance(a[path], (bool, str)):
                if a[path] != b[path]:
                    raise GateFailure(f"A/A discrete mismatch at step {step}, {path}")
                continue
            if isinstance(a[path], (int, float)) and isinstance(b[path], (int, float)):
                compared += 1
                left_value, right_value = float(a[path]), float(b[path])
                if left_value != right_value:
                    absolute = abs(left_value - right_value)
                    relative = absolute / max(abs(left_value), abs(right_value), 1e-30)
                    differences.append({
                        "step": step, "path": path, "reference": left_value, "repeat": right_value,
                        "absolute_difference": absolute, "relative_difference": relative,
                    })
    differences.sort(key=lambda item: (item["relative_difference"], item["absolute_difference"]), reverse=True)
    return {
        "discrete_trajectory_exact": True,
        "numeric_trajectory_exact": not differences,
        "compared_numeric_values": compared,
        "differing_numeric_values": len(differences),
        "maximum_absolute_difference": max((item["absolute_difference"] for item in differences), default=0.0),
        "maximum_relative_difference": max((item["relative_difference"] for item in differences), default=0.0),
        "largest_differences": differences[:25],
    }


def checkpoint_digest(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise GateFailure(f"Missing checkpoint: {root}")
    digest = hashlib.sha256(); count = size = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix(); file_hash = base.sha256(path)
        byte_size = path.stat().st_size; count += 1; size += byte_size
        digest.update(f"{relative}\0{byte_size}\0{file_hash}\n".encode())
    return {"path": str(root), "file_count": count, "byte_size": size, "tree_sha256": digest.hexdigest()}


def verify_checkpoints(uv: str, worktree: Path, evidence: Path, root: Path, label: str, env: Mapping[str, str]) -> dict[str, Any]:
    code = (
        "import json,sys; from pathlib import Path; "
        "from dataset.src.joint_checkpoint import verify_local_manifest; "
        "roots=[Path(x) for x in sys.argv[1:]]; "
        "out={str(x):len(verify_local_manifest(x)['files']) for x in roots}; "
        "print(json.dumps(out,sort_keys=True))"
    )
    return base.run(
        [uv, "run", "--python", "3.13", "--extra", "model", "python", "-c", code,
         str(root / "step-00000025"), str(root / "step-00000050")],
        name=f"verify-checkpoints-{label}", evidence=evidence, cwd=worktree, env=env,
    )


def trainer_command(uv: str, dataset: Path, checkpoints: Path, run_id: str, name: str, entity: str | None) -> list[str]:
    command = [
        uv, "run", "--python", "3.13", "--extra", "model", "--with", "wandb==0.26.1",
        "--with-requirements", "dataset/requirements-remote.txt", "python", "-m", "trainer",
        "--dataset-dir", str(dataset), "--dataset-manifest", str(dataset / "manifest.json"),
        "--checkpoint-dir", str(checkpoints), "--steps", "50", "--sequences-per-block", "16",
        "--model-size", "smoke", "--architecture", "gdn2_hybrid", "--gdn-chunk-size", "32",
        "--initialization", "normal", "--optimizer", "hybrid_muon_adamw", "--device", "cuda",
        "--precision", "fp16", "--microbatch-size", "1", "--learning-rate", "3e-4",
        "--weight-decay", "0.1", "--muon-momentum", "0.95", "--muon-lr-multiplier", "1.0",
        "--muon-update-rms", "0.18", "--muon-weight-decay", "0.1", "--max-grad-norm", "1.0",
        "--schedule", "wsd", "--warmup-tokens", "524288", "--stable-tokens", "7471104",
        "--decay-tokens", "2011136", "--minimum-lr-ratio", "0.1",
        "--checkpoint-every-steps", "25", "--evaluation-every-steps", "50",
        "--validation-blocks", "1", "--seed", "17", "--wandb-mode", "online",
        "--wandb-project", "Small-LLM", "--wandb-run-id", run_id, "--wandb-run-name", name,
        "--wandb-resume", "never", "--wandb-tags", "20m", "t4", "repeatability", "aa",
    ]
    if entity:
        command.extend(["--wandb-entity", entity])
    return command


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run two exact 50-update WSD prefixes and compare them")
    parser.add_argument("--launch-commit", default=os.environ.get("SMALL_LLM_LAUNCH_COMMIT", COMMIT))
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--wandb-run-prefix", default=os.environ.get("SMALL_LLM_WANDB_AA_PREFIX"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = arguments(argv); WORK.mkdir(parents=True, exist_ok=True)
    evidence = WORK / f"small-llm-repeatability-{stamp()}"; evidence.mkdir()
    summary: dict[str, Any] = {
        "schema_version": 1, "started_utc": utc_now(), "status": "running",
        "authorization": "none", "evidence": str(evidence),
        "scope": "same-T4 50-update reference and A/A WSD-prefix repeatability measurement",
        "stages": [],
    }
    try:
        summary["environment"] = base.check_environment()
        wandb_key = base.secret("WANDB_API_KEY"); entity = base.secret("WANDB_ENTITY", required=False)
        if not wandb_key:
            raise GateFailure("Missing required Kaggle Secret: WANDB_API_KEY")
        worktree, checkout, stage = prepare_worktree(args.launch_commit, evidence)
        summary["checkout"] = checkout; summary["stages"].append(stage)
        env = {"WANDB_API_KEY": wandb_key, "UV_LINK_MODE": "copy"}
        summary["stages"].append(base.run(
            [sys.executable, "-m", "pip", "install", "-q", "uv"],
            name="install-uv", evidence=evidence,
        ))
        uv = base.find_uv()
        summary["stages"].append(base.run(
            [uv, "python", "install", "3.13"], name="install-python-3.13",
            evidence=evidence, cwd=worktree, env=env,
        ))
        summary["stages"].append(base.run(
            [uv, "run", "--python", "3.13", "python", "--version"],
            name="python-version", evidence=evidence, cwd=worktree, env=env,
        ))
        dataset, inspected = base.find_dataset(args.dataset_dir)
        summary["dataset"] = {"path": str(dataset), "inspected": inspected}
        summary["stages"].append(base.run(
            [uv, "run", "--python", "3.13", "--with-requirements", "dataset/requirements-remote.txt",
             "python", "-m", "dataset.main", "verify", "--output-dir", str(dataset), "--full-scan"],
            name="dataset-full-scan", evidence=evidence, cwd=worktree, env=env,
        ))
        plan_path = evidence / "qualification_plan.json"
        summary["stages"].append(base.run(
            [uv, "run", "--python", "3.13", "python", "-m", "dataset.qualification_20m_report",
             "--dataset-dir", str(dataset), "--drive-manifest", str(dataset / "drive_manifest.json"),
             "--output", str(plan_path)],
            name="qualification-plan", evidence=evidence, cwd=worktree, env=env,
        ))
        summary["plan"] = base.validate_plan(plan_path)

        prefix = args.wandb_run_prefix or f"20m-t4-aa-{run_id_stamp()}"
        parsed: dict[str, Any] = {}; runs: dict[str, Any] = {}; trees: dict[str, Any] = {}
        for label, run_id, run_name in (
            ("reference", f"{prefix}-reference", "20M T4 reference 50"),
            ("repeat", f"{prefix}-repeat", "20M T4 A-A repeat 50"),
        ):
            checkpoints = evidence / label / "checkpoints"
            training = base.run(
                trainer_command(uv, dataset, checkpoints, run_id, run_name, entity),
                name=f"trainer-{label}-50", evidence=evidence, cwd=worktree, env=env,
            )
            summary["stages"].append(training)
            summary["stages"].append(verify_checkpoints(uv, worktree, evidence, checkpoints, label, env))
            parsed[label] = parse_log(Path(training["log"]))
            runs[label] = {"wandb_run_id": run_id, "checkpoint_dir": str(checkpoints), "metrics": summarize(parsed[label])}
            trees[label] = {
                f"step_{step}": checkpoint_digest(checkpoints / f"step-{step:08d}")
                for step in (25, 50)
            }

        metric_comparison = compare_rows(parsed["reference"]["rows"], parsed["repeat"]["rows"])
        validation_exact = parsed["reference"]["validation"] == parsed["repeat"]["validation"]
        tree_exact = {
            f"step_{step}": trees["reference"][f"step_{step}"]["tree_sha256"]
            == trees["repeat"][f"step_{step}"]["tree_sha256"]
            for step in (25, 50)
        }
        bitwise = metric_comparison["numeric_trajectory_exact"] and validation_exact and all(tree_exact.values())
        summary["runs"] = runs; summary["checkpoint_trees"] = trees
        summary["comparison"] = {
            "metric_trajectory": metric_comparison,
            "validation_exact": validation_exact,
            "checkpoint_tree_exact": tree_exact,
            "repeatability_class": "bitwise_identical" if bitwise else "measured_nondeterminism_requires_threshold_review",
        }
        summary["review_flags"] = {
            "clipping_frequency_above_provisional_failure_band": any(
                runs[label]["metrics"]["clipping"]["fraction"] > 0.5 for label in runs
            ),
            "last_10_gradient_median_above_first_10": any(
                runs[label]["metrics"]["gradient_norm"]["last_10_median"]
                > runs[label]["metrics"]["gradient_norm"]["first_10_median"] for label in runs
            ),
            "checkpoint_or_metric_threshold_review_required": not bitwise,
        }
        summary["status"] = "passed_repeatability_measurement"
        summary["authorization"] = "threshold_review_only"
        summary["finished_utc"] = utc_now()
        base.write_json(evidence / "repeatability_summary.json", summary); base.write_json(LATEST, summary)
        print(json.dumps({
            "status": summary["status"], "authorization": summary["authorization"],
            "repeatability_class": summary["comparison"]["repeatability_class"],
            "review_flags": summary["review_flags"], "summary": str(LATEST), "evidence": str(evidence),
        }, indent=2, sort_keys=True))
        return 0
    except BaseException as error:
        summary.update({
            "status": "failed", "authorization": "none", "finished_utc": utc_now(),
            "error_type": type(error).__name__, "error": str(error),
        })
        base.write_json(evidence / "repeatability_summary.json", summary); base.write_json(LATEST, summary)
        print(json.dumps({
            "status": "failed", "error_type": type(error).__name__, "error": str(error),
            "summary": str(LATEST), "evidence": str(evidence),
        }, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
