"""Provider-neutral execution logic used inside the Modal GPU function."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from profiles import (
    DEFAULT_PRECISION,
    DURABILITY_EVERY,
    MICROBATCH_CANDIDATES,
    SEQUENCES_PER_BLOCK,
    ModelPreset,
    TokenPreset,
    canonical_run_id,
    resolve_presets,
    run_name,
)

PROBE_STEPS = 4
PROBE_WARMUP = 1
MAX_RESERVED_MEMORY_FRACTION = 0.90
_CHECKPOINT_ID = re.compile(r"^step-(\d{8})$")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError(f"expected JSON object: {path}")
    return dict(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: Sequence[str], *, cwd: Path, log_path: Path, check: bool = True) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$ " + " ".join(str(x) for x in command), flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
        code = process.wait()
    if check and code:
        raise RuntimeError(f"command failed with exit code {code}; see {log_path}")
    return code


def _dataset_matches(root: Path, profile_key: str) -> tuple[bool, dict[str, Any]]:
    from dataset.qualification import get_profile

    contract = get_profile(profile_key)
    manifest_path, drive_path = root / "manifest.json", root / "drive_manifest.json"
    row: dict[str, Any] = {
        "root": str(root),
        "manifest": manifest_path.is_file(),
        "drive_manifest": drive_path.is_file(),
        "train": (root / "train").is_dir(),
        "validation": (root / "validation").is_dir(),
    }
    if not all(row[key] for key in ("manifest", "drive_manifest", "train", "validation")):
        return False, row
    manifest = _json(manifest_path)
    production = manifest.get("production")
    expected_top = {
        "schema_version": 2,
        "sequence_format": "context_plus_one",
        "context_length": contract.context_length,
        "stored_tokens_per_sequence": contract.context_length + 1,
        "sequences_per_block": contract.sequences_per_block,
        "target_shard_bytes": contract.target_shard_bytes,
    }
    expected_prod = {
        "run_id": contract.run_id,
        "target_source_tokens": contract.target_source_tokens,
        "minimum_source_tokens": contract.minimum_source_tokens,
        "maximum_source_tokens": contract.maximum_source_tokens,
        "checkpoint_source_tokens": contract.checkpoint_source_tokens,
        "target_reached": True,
        "remote_required": True,
    }
    matched = all(manifest.get(key) == value for key, value in expected_top.items())
    matched = matched and isinstance(production, Mapping)
    if isinstance(production, Mapping):
        matched = matched and all(production.get(key) == value for key, value in expected_prod.items())
        row["run_id"] = production.get("run_id")
    return bool(matched), row


def _find_dataset(data_root: Path, explicit: str, profile_key: str) -> tuple[Path, list[dict[str, Any]]]:
    if explicit:
        candidate = Path(explicit)
        if not candidate.is_absolute():
            candidate = data_root / candidate
        roots = [candidate.resolve()]
    else:
        roots = sorted({path.parent for path in data_root.rglob("manifest.json")})
    inspected: list[dict[str, Any]] = []
    matches: list[Path] = []
    for root in roots:
        matched, row = _dataset_matches(root, profile_key)
        inspected.append(row)
        if matched:
            matches.append(root)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one Modal dataset matching {profile_key}; found {len(matches)}\n"
            + json.dumps(inspected, indent=2)
        )
    return matches[0], inspected


def _derive_plan(repo_root: Path, dataset: Path, profile_key: str, output: Path, log_path: Path) -> dict[str, Any]:
    _run(
        [
            sys.executable, "-m", "dataset.qualification", "report",
            "--profile", profile_key,
            "--dataset-dir", str(dataset),
            "--drive-manifest", str(dataset / "drive_manifest.json"),
            "--output", str(output),
        ],
        cwd=repo_root,
        log_path=log_path,
    )
    plan = _json(output)
    trainer = plan.get("trainer")
    if not isinstance(trainer, Mapping):
        raise RuntimeError("qualification plan has no trainer section")
    if trainer.get("full_block_target_tokens") != SEQUENCES_PER_BLOCK * 2048:
        raise RuntimeError("qualification plan changed the frozen 16-sequence optimizer block")
    return plan


def _trainer_command(
    *,
    model: ModelPreset,
    tokens: TokenPreset,
    dataset: Path,
    plan: Mapping[str, Any],
    checkpoint_dir: Path,
    steps: int,
    microbatch: int,
    precision: str,
    wandb_run_id: str,
    gpu_tag: str,
    online: bool,
    resume: str | None = None,
) -> list[str]:
    trainer = plan["trainer"]
    assert isinstance(trainer, Mapping)
    command = [
        sys.executable, "-m", "trainer",
        "--dataset-dir", str(dataset),
        "--dataset-manifest", str(dataset / "manifest.json"),
        "--checkpoint-dir", str(checkpoint_dir),
        "--steps", str(steps),
        "--sequences-per-block", str(SEQUENCES_PER_BLOCK),
        "--model-size", model.trainer_size,
        "--architecture", "gdn2_hybrid",
        "--gdn-chunk-size", "32",
        "--initialization", "normal",
        "--optimizer", "hybrid_muon_adamw",
        "--device", "cuda",
        "--precision", precision,
        "--microbatch-size", str(microbatch),
        "--learning-rate", "3e-4",
        "--weight-decay", "0.1",
        "--muon-momentum", "0.95",
        "--muon-lr-multiplier", "1.0",
        "--muon-update-rms", "0.18",
        "--muon-weight-decay", "0.1",
        "--max-grad-norm", "1.0",
        "--schedule", "wsd",
        "--warmup-tokens", str(trainer["warmup_tokens"]),
        "--stable-tokens", str(trainer["stable_tokens"]),
        "--decay-tokens", str(trainer["decay_tokens"]),
        "--minimum-lr-ratio", "0.1",
        "--seed", "17",
    ]
    if resume:
        command += ["--resume", resume]
    if not online:
        return command + [
            "--checkpoint-every-steps", "0",
            "--evaluation-every-steps", "0",
            "--validation-blocks", "0",
            "--remote-publish-every-steps", "0",
            "--wandb-mode", "disabled",
        ]
    command += [
        "--checkpoint-every-steps", str(DURABILITY_EVERY),
        "--evaluation-every-steps", str(DURABILITY_EVERY),
        "--validation-blocks", str(trainer["validation_blocks"]),
        # Modal Volumes are the durable checkpoint transport here. The legacy
        # HF protocol namespaces checkpoints by dataset run ID, which collides
        # if this same finite corpus is reused by a different model size.
        "--remote-publish-every-steps", "0",
        "--wandb-mode", "online",
        "--wandb-project", "Small-LLM",
        "--wandb-run-id", wandb_run_id,
        "--wandb-run-name", run_name(model, tokens),
        "--wandb-tags",
        model.label.lower(), f"{tokens.label.lower()}-tokens", "modal", gpu_tag,
        "data-scaling", f"microbatch-{microbatch}", "one-pass", "exact-resume",
        "--wandb-resume", "must" if resume else "allow",
    ]
    entity = os.environ.get("WANDB_ENTITY")
    if entity:
        command += ["--wandb-entity", entity]
    return command


def _training_rows(log_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(item, Mapping) and all(
            key in item for key in ("step", "loss", "gradient_norm", "tokens_per_second")
        ):
            rows.append(dict(item))
    return rows


def _gpu_environment() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Modal function has no CUDA device")
    props = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    return {
        "name": props.name,
        "total_memory_bytes": int(props.total_memory),
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }


def _qualify_microbatch(
    *,
    repo_root: Path,
    model: ModelPreset,
    tokens: TokenPreset,
    dataset: Path,
    plan: Mapping[str, Any],
    run_dir: Path,
    precision: str,
    requested: int,
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    candidates = (requested,) if requested else MICROBATCH_CANDIDATES
    total_bytes = int(environment["total_memory_bytes"])
    results: dict[str, Any] = {}
    for candidate in candidates:
        probe_dir = Path("/tmp") / f"small-llm-probe-{candidate}"
        shutil.rmtree(probe_dir, ignore_errors=True)
        log_path = run_dir / "evidence" / f"microbatch-{candidate}-probe.log"
        code = _run(
            _trainer_command(
                model=model,
                tokens=tokens,
                dataset=dataset,
                plan=plan,
                checkpoint_dir=probe_dir,
                steps=PROBE_STEPS,
                microbatch=candidate,
                precision=precision,
                wandb_run_id="probe-disabled",
                gpu_tag="probe",
                online=False,
            ),
            cwd=repo_root,
            log_path=log_path,
            check=False,
        )
        rows = _training_rows(log_path)
        row: dict[str, Any] = {"exit_code": code, "rows": len(rows), "safe": False}
        if code == 0 and len(rows) == PROBE_STEPS:
            measured = rows[PROBE_WARMUP:]
            tps = [float(item["tokens_per_second"]) for item in measured]
            loss, grad = float(rows[-1]["loss"]), float(rows[-1]["gradient_norm"])
            peak = max(int(item.get("peak_reserved_memory_bytes", 0)) for item in rows)
            fraction = peak / total_bytes
            finite = math.isfinite(loss) and math.isfinite(grad) and all(math.isfinite(x) for x in tps)
            row.update(
                median_tokens_per_second=statistics.median(tps),
                final_loss=loss,
                final_gradient_norm=grad,
                peak_reserved_memory_bytes=peak,
                reserved_memory_fraction=fraction,
                finite=finite,
                safe=bool(finite and fraction <= MAX_RESERVED_MEMORY_FRACTION),
            )
        results[str(candidate)] = row
        shutil.rmtree(probe_dir, ignore_errors=True)
    safe = [(int(key), row) for key, row in results.items() if row.get("safe") is True]
    if not safe:
        raise RuntimeError("no microbatch passed Modal GPU qualification: " + json.dumps(results, indent=2))
    selected, selected_row = max(safe, key=lambda item: float(item[1]["median_tokens_per_second"]))
    return {
        "status": "passed",
        "selected_microbatch": selected,
        "selection": "fastest_safe_measured_candidate",
        "maximum_reserved_memory_fraction": MAX_RESERVED_MEMORY_FRACTION,
        "results": results,
        "selected_median_tokens_per_second": selected_row["median_tokens_per_second"],
        "gpu": dict(environment),
    }


def _latest_checkpoint(checkpoint_dir: Path) -> tuple[str | None, int]:
    from dataset.src.joint_checkpoint import verify_local_manifest

    if not checkpoint_dir.is_dir():
        return None, 0
    valid: list[tuple[int, str]] = []
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
                valid.append((step, root.name))
        except Exception as error:  # noqa: BLE001 - never load an invalid candidate
            print(f"Ignoring invalid checkpoint {root}: {type(error).__name__}: {error}", flush=True)
    if not valid:
        return None, 0
    step, checkpoint_id = max(valid)
    return checkpoint_id, step


def _runtime_contract(
    *,
    source_commit: str,
    model: ModelPreset,
    tokens: TokenPreset,
    precision: str,
    microbatch: int,
    dataset: Path,
    environment: Mapping[str, Any],
    qualification: object,
) -> dict[str, Any]:
    manifest = _json(dataset / "manifest.json")
    production = manifest.get("production")
    return {
        "version": 1,
        "source_commit": source_commit,
        "model_parameters": model.parameters,
        "model_label": model.label,
        "model_size": model.trainer_size,
        "training_tokens": tokens.tokens,
        "token_label": tokens.label,
        "dataset_profile": tokens.dataset_profile,
        "dataset_run_id": production.get("run_id") if isinstance(production, Mapping) else None,
        "precision": precision,
        "microbatch_size": microbatch,
        "sequences_per_block": SEQUENCES_PER_BLOCK,
        "architecture": "gdn2_hybrid",
        "gdn_chunk_size_saved": 32,
        "fla_internal_chunk_size": 64,
        "optimizer": "hybrid_muon_adamw",
        "seed": 17,
        "gpu_first_qualified": dict(environment),
        "microbatch_qualification": qualification,
    }


def _assert_contract(path: Path, expected: Mapping[str, Any]) -> None:
    if not path.is_file():
        _write_json(path, expected)
        return
    actual = _json(path)
    immutable = (
        "source_commit", "model_parameters", "model_size", "training_tokens",
        "dataset_profile", "dataset_run_id", "precision", "microbatch_size",
        "sequences_per_block", "architecture", "gdn_chunk_size_saved", "optimizer", "seed",
    )
    drift = {key: (actual.get(key), expected.get(key)) for key in immutable if actual.get(key) != expected.get(key)}
    if drift:
        raise RuntimeError("refusing Modal run configuration drift: " + json.dumps(drift, indent=2))


def run_training(
    *,
    model: str,
    tokens: str,
    source_commit: str,
    dataset_dir: str,
    max_steps_this_session: int,
    microbatch_size: int,
    precision: str,
    repo_root: Path,
    data_root: Path,
    run_root: Path,
    cache_root: Path,
    run_volume: object,
    cache_volume: object,
) -> dict[str, object]:
    os.chdir(repo_root)
    if precision != DEFAULT_PRECISION:
        raise RuntimeError("the first Modal production migration is frozen to fp16")
    if not 0 <= microbatch_size <= SEQUENCES_PER_BLOCK:
        raise RuntimeError(f"microbatch-size must be 0 (auto) or 1..{SEQUENCES_PER_BLOCK}")
    if max_steps_this_session < 0:
        raise RuntimeError("max-steps-this-session cannot be negative")

    model_preset, token_preset = resolve_presets(model, tokens)
    run_id = canonical_run_id(model_preset, token_preset)
    run_dir = run_root / run_id
    checkpoint_dir = run_dir / "checkpoints"
    evidence_dir = run_dir / "evidence"
    plan_path = run_dir / "qualification_plan.json"
    runtime_path = run_dir / "modal_runtime.json"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    environment = _gpu_environment()
    capability = str(environment["compute_capability"]).replace(".", "")
    triton_cache = cache_root / "triton" / f"sm{capability}" / "torch-2.10-triton-3.6"
    triton_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(triton_cache)

    dataset, inspected = _find_dataset(data_root, dataset_dir, token_preset.dataset_profile)
    identity = {
        "dataset": str(dataset),
        "manifest_sha256": _sha256(dataset / "manifest.json"),
        "drive_manifest_sha256": _sha256(dataset / "drive_manifest.json"),
    }
    verify_marker = run_dir / "dataset_verified.json"
    marker = _json(verify_marker) if verify_marker.is_file() else {}
    if marker.get("identity") != identity:
        _run(
            [sys.executable, "-m", "dataset.main", "verify", "--output-dir", str(dataset), "--full-scan"],
            cwd=repo_root,
            log_path=evidence_dir / "dataset-verify.log",
        )
        _write_json(verify_marker, {"identity": identity, "datasets_inspected": inspected})

    plan = _derive_plan(
        repo_root, dataset, token_preset.dataset_profile, plan_path,
        evidence_dir / "qualification-plan.log",
    )
    latest_id, completed = _latest_checkpoint(checkpoint_dir)

    if runtime_path.is_file():
        existing = _json(runtime_path)
        frozen_microbatch = existing.get("microbatch_size")
        if not isinstance(frozen_microbatch, int):
            raise RuntimeError("saved Modal runtime has no valid microbatch_size")
        if microbatch_size and microbatch_size != frozen_microbatch:
            raise RuntimeError(f"run froze microbatch {frozen_microbatch}; requested {microbatch_size}")
        selected_microbatch = frozen_microbatch
        qualification = existing.get("microbatch_qualification")
    else:
        qualification = _qualify_microbatch(
            repo_root=repo_root,
            model=model_preset,
            tokens=token_preset,
            dataset=dataset,
            plan=plan,
            run_dir=run_dir,
            precision=precision,
            requested=microbatch_size,
            environment=environment,
        )
        selected_microbatch = int(qualification["selected_microbatch"])

    contract = _runtime_contract(
        source_commit=source_commit,
        model=model_preset,
        tokens=token_preset,
        precision=precision,
        microbatch=selected_microbatch,
        dataset=dataset,
        environment=environment,
        qualification=qualification,
    )
    _assert_contract(runtime_path, contract)
    if not latest_id:
        _write_json(runtime_path, contract)
    getattr(run_volume, "commit")()
    getattr(cache_volume, "commit")()

    trainer = plan["trainer"]
    assert isinstance(trainer, Mapping)
    total_steps = int(trainer["steps"])
    if completed > total_steps:
        raise RuntimeError(f"checkpoint step {completed} exceeds plan total {total_steps}")
    remaining = total_steps - completed
    if remaining == 0:
        return {
            "status": "already_complete",
            "run_id": run_id,
            "completed_steps": completed,
            "total_steps": total_steps,
            "microbatch_size": selected_microbatch,
            "gpu": environment,
        }

    additional = remaining if max_steps_this_session == 0 else min(remaining, max_steps_this_session)
    gpu_tag = re.sub(r"[^a-z0-9]+", "-", str(environment["name"]).lower()).strip("-")
    log_path = evidence_dir / f"train-from-{completed:08d}.log"
    started = time.perf_counter()
    _run(
        _trainer_command(
            model=model_preset,
            tokens=token_preset,
            dataset=dataset,
            plan=plan,
            checkpoint_dir=checkpoint_dir,
            steps=additional,
            microbatch=selected_microbatch,
            precision=precision,
            wandb_run_id=run_id,
            gpu_tag=gpu_tag,
            online=True,
            resume=latest_id,
        ),
        cwd=repo_root,
        log_path=log_path,
    )
    getattr(run_volume, "commit")()
    getattr(cache_volume, "commit")()

    final_id, final_step = _latest_checkpoint(checkpoint_dir)
    expected = completed + additional
    if final_step != expected:
        raise RuntimeError(f"durable checkpoint step {final_step} != expected {expected}")
    result = {
        "status": "complete" if final_step == total_steps else "segment_complete",
        "run_id": run_id,
        "checkpoint_id": final_id,
        "completed_steps": final_step,
        "total_steps": total_steps,
        "elapsed_seconds": time.perf_counter() - started,
        "microbatch_size": selected_microbatch,
        "gpu": environment,
        "dataset": str(dataset),
        "source_commit": source_commit,
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result
