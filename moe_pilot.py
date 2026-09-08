"""Provider-neutral 100M/2B pilot contract and runner.

This module lives outside the training package, so CPU preparation does not
execute ``trainer.__init__`` or import a training framework.  The child itself
is started with ``python -m trainer`` or ``python -m MOE_model``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
import time

from dataset.qualification import get_profile
from dataset.qualification_report import derive_plan
from dataset.src.joint_checkpoint import verify_local_manifest
from dataset.src.storage import read_json, write_json_atomic
from dataset.src.verify import verify


PILOT_MODEL, PILOT_TOKENS, PILOT_PROFILE = "100M", "2B", "modal-2b-b64"
PILOT_ARMS = ("D", "M0", "M1")
DEFAULT_CHECKPOINT_STEPS = (0, 25, 50, 100, 200, 300, 500, 750, 1000)
DEFAULT_PROFILE_STEPS = (50, 400, 900)
_SAFE_LEAF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
_CHECKPOINT_ID = re.compile(r"^step-(\d{1,12})$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_leaf(value: str, label: str = "value") -> str:
    if not isinstance(value, str) or _SAFE_LEAF.fullmatch(value) is None:
        raise ValueError(f"{label} must be a safe single path component")
    return value


def _positive(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def _sha(value: object) -> str:
    if not isinstance(value, str) or _FULL_SHA.fullmatch(value) is None:
        raise ValueError("source_commit must be a full lowercase Git SHA")
    return value


def _gamma(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("gamma must be finite and non-negative")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("gamma must be finite and non-negative")
    return value


def _steps(value: str) -> tuple[int, ...]:
    try:
        result = tuple(sorted({int(item) for item in value.split(",")}))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated update numbers") from error
    if not result or result[0] < 0:
        raise argparse.ArgumentTypeError("update numbers must be non-negative")
    return result


@dataclass(frozen=True, slots=True)
class PilotRequest:
    provider: str
    arm: str
    run_id: str
    dataset_dir: Path
    run_root: Path
    steps: int
    precision: str
    microbatch_size: int
    source_commit: str
    seed: int = 17
    gamma: float | None = None
    probe_sequences: int = 2
    probe_lm_logits: bool = False
    checkpoint_steps: tuple[int, ...] | None = None
    profile_steps: tuple[int, ...] | None = None


@dataclass(frozen=True, slots=True)
class PilotPlan:
    manifest_path: Path
    manifest_sha256: str
    trainer: Mapping[str, object]
    steps: int
    sequences_per_block: int


@dataclass(frozen=True, slots=True)
class PilotSpec:
    request: PilotRequest
    plan: PilotPlan
    namespace: Path
    checkpoint_dir: Path
    experiment_dir: Path
    checkpoint_steps: tuple[int, ...]
    profile_steps: tuple[int, ...]


def validate_request(request: PilotRequest) -> None:
    if request.provider not in {"modal", "beam"}:
        raise ValueError("provider must be modal or beam")
    if request.arm not in PILOT_ARMS:
        raise ValueError(f"arm must be one of {PILOT_ARMS}")
    _safe_leaf(request.run_id, "run_id")
    _sha(request.source_commit)
    _positive(request.steps, "steps")
    max_steps = 100 if request.arm == "D" else (300 if request.provider == "modal" else 1000)
    if request.steps > max_steps:
        raise ValueError(f"{request.provider} {request.arm} pilot cap exceeds {max_steps} updates")
    if request.precision not in {"fp16", "bf16", "fp32"}:
        raise ValueError("precision must be fp16, bf16 or fp32")
    if _positive(request.microbatch_size, "microbatch_size") > 64:
        raise ValueError("microbatch_size cannot exceed the block size of 64")
    if isinstance(request.seed, bool) or not isinstance(request.seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(request.probe_sequences, bool) or request.probe_sequences < 0:
        raise ValueError("probe_sequences cannot be negative")
    gamma = None if request.gamma is None else _gamma(request.gamma)
    if request.arm == "M1" and gamma is None:
        raise ValueError("M1 requires an explicit gamma")
    if request.arm != "M1" and gamma not in (None, 0.0):
        raise ValueError("only M1 accepts a non-zero gamma")
    if not isinstance(request.probe_lm_logits, bool):
        raise ValueError("probe_lm_logits must be boolean")


def _selected(cap: int, supplied: tuple[int, ...] | None, profile: bool) -> tuple[int, ...]:
    if supplied is None:
        values = (
            tuple(step for step in DEFAULT_PROFILE_STEPS if step <= cap)
            if profile else {step for step in DEFAULT_CHECKPOINT_STEPS if step <= cap}
        )
    else:
        values = set(supplied)
    if any(isinstance(step, bool) or not isinstance(step, int) for step in values):
        raise ValueError("selected update numbers must be integers")
    if any(step < 0 or step > cap for step in values):
        raise ValueError("selected update numbers must lie within the absolute step cap")
    if profile and 0 in values:
        raise ValueError("profile update numbers must be positive")
    if profile:
        return tuple(sorted(values))
    return tuple(sorted(set(values) | {0, cap}))


def _plan_from_manifest(dataset_dir: Path) -> PilotPlan:
    root = dataset_dir.resolve()
    manifest_path = root / "manifest.json"
    if not root.is_dir() or not manifest_path.is_file():
        raise FileNotFoundError(f"prepared schema-v2 manifest does not exist: {manifest_path}")
    report = verify(root, full_scan=False)
    if not report.passed or not report.complete:
        raise RuntimeError("prepared dataset failed verification: " + "; ".join(report.problems))
    manifest = read_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise ValueError("prepared dataset manifest must be an object")
    plan = derive_plan(manifest, profile=get_profile(PILOT_PROFILE).plan, manifest_path=manifest_path)
    trainer = plan.get("trainer")
    required = {"steps", "full_block_target_tokens", "schedule", "warmup_tokens",
                "stable_tokens", "decay_tokens", "minimum_lr_ratio", "validation_blocks"}
    if not isinstance(trainer, Mapping) or trainer.get("schedule") != "wsd" or not required <= set(trainer):
        raise RuntimeError("prepared dataset did not produce the complete frozen WSD plan")
    return PilotPlan(
        manifest_path=manifest_path,
        manifest_sha256=_sha256(manifest_path),
        trainer=dict(trainer),
        steps=_positive(trainer["steps"], "qualification steps"),
        sequences_per_block=_positive(plan["sequences_per_block"], "sequences_per_block"),
    )


def _identity(request: PilotRequest, plan: PilotPlan, namespace: Path,
              checkpoints: tuple[int, ...], profiles: tuple[int, ...]) -> dict[str, object]:
    return {
        "version": 1, "provider": request.provider, "run_id": request.run_id,
        "namespace": str(namespace), "arm": request.arm, "model": PILOT_MODEL,
        "tokens": PILOT_TOKENS, "source_commit": request.source_commit,
        "gamma": 0.0 if request.gamma is None else float(request.gamma),
        "dataset_profile": PILOT_PROFILE, "dataset_manifest_sha256": plan.manifest_sha256,
        "precision": request.precision, "microbatch_size": request.microbatch_size,
        "seed": request.seed, "schedule": dict(plan.trainer), "steps_cap": request.steps,
        "checkpoint_steps": list(checkpoints), "profile_steps": list(profiles),
        "probe_sequences": request.probe_sequences, "probe_lm_logits": request.probe_lm_logits,
    }


def prepare_pilot(request: PilotRequest) -> PilotSpec:
    validate_request(request)
    plan = _plan_from_manifest(request.dataset_dir)
    if request.steps > plan.steps:
        raise ValueError(f"pilot cap {request.steps} exceeds manifest schedule {plan.steps}")
    checkpoints = _selected(request.steps, request.checkpoint_steps, False)
    profiles = _selected(request.steps, request.profile_steps, True)
    run_root = request.run_root.resolve()
    namespace = (run_root / "moe-pilots" / request.provider /
                 _safe_leaf(request.run_id, "run_id") / request.arm)
    try:
        namespace.resolve().relative_to(run_root)
    except ValueError as error:
        raise RuntimeError("pilot namespace escapes the mounted run volume") from error
    checkpoint_dir, experiment_dir = namespace / "checkpoints", namespace / "experiment"
    namespace.mkdir(parents=True, exist_ok=True)
    if checkpoint_dir.is_symlink() or experiment_dir.is_symlink():
        raise RuntimeError("pilot output directories cannot be symlinks")
    contract = namespace / "pilot_contract.json"
    expected = _identity(request, plan, namespace, checkpoints, profiles)
    if contract.is_symlink():
        raise RuntimeError("pilot contract cannot be a symlink")
    if contract.exists():
        actual = read_json(contract)
        if not isinstance(actual, Mapping) or actual.get("identity") != expected:
            raise RuntimeError("pilot namespace belongs to a different frozen experiment identity")
    else:
        if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
            raise RuntimeError("pilot checkpoints exist without a frozen pilot contract")
        write_json_atomic(contract, {"identity": expected, "plan": {"trainer": dict(plan.trainer)}})
    return PilotSpec(request, plan, namespace, checkpoint_dir, experiment_dir, checkpoints, profiles)


def _step(path: Path) -> int | None:
    match = _CHECKPOINT_ID.fullmatch(path.name)
    return None if match is None else int(match.group(1))


def _complete_checkpoint(path: Path) -> dict[str, object]:
    step = _step(path)
    if step is None or path.is_symlink() or not path.is_dir():
        raise ValueError("invalid checkpoint directory")
    verify_local_manifest(path)
    payload = read_json(path / "checkpoint.json")
    pipeline = payload.get("pipeline_state") if isinstance(payload, Mapping) else None
    if (not isinstance(payload, Mapping) or payload.get("checkpoint_id") != path.name or
        payload.get("optimizer_step_complete") is not True or not isinstance(pipeline, Mapping) or
        pipeline.get("gradient_accumulation_position", 0) != 0 or
        pipeline.get("last_consumed_block_id") != step - 1):
        raise ValueError("checkpoint metadata does not describe this completed update")
    with (path / "trainer_state.pkl").open("rb") as handle:
        state = pickle.load(handle)
    if not isinstance(state, Mapping) or state.get("global_step") != step:
        raise ValueError("checkpoint state step differs from its identity")
    tokens = state.get("consumed_tokens")
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
        raise ValueError("invalid checkpoint target count")
    return {"checkpoint_id": path.name, "step": step, "path": path, "consumed_tokens": tokens}


def find_latest_complete_checkpoint(checkpoint_dir: Path) -> dict[str, object] | None:
    if not checkpoint_dir.is_dir() or checkpoint_dir.is_symlink():
        return None
    candidates = sorted((p for p in checkpoint_dir.iterdir() if p.is_dir()),
                        key=lambda p: _step(p) if _step(p) is not None else -1, reverse=True)
    for path in candidates:
        try:
            return _complete_checkpoint(path)
        except (OSError, EOFError, ValueError, TypeError, AttributeError, ImportError,
                IndexError, pickle.PickleError, RuntimeError):
            continue
    return None


def build_pilot_command(spec: PilotSpec, *, remaining_steps: int, resume: str | None,
                        python_executable: str = sys.executable) -> list[str]:
    if remaining_steps <= 0:
        raise ValueError("remaining_steps must be positive")
    if resume is not None and _CHECKPOINT_ID.fullmatch(resume) is None:
        raise ValueError("resume must be a complete step checkpoint ID")
    req, plan = spec.request, spec.plan
    previous = int(resume.split("-", 1)[1]) if resume else 0
    if remaining_steps != req.steps - previous:
        raise ValueError("remaining steps must end at the frozen absolute cap")
    t = plan.trainer
    command = [
        python_executable, "-m", "trainer" if req.arm == "D" else "MOE_model",
        "--dataset-dir", str(req.dataset_dir.resolve()), "--dataset-manifest", str(plan.manifest_path),
        "--checkpoint-dir", str(spec.checkpoint_dir), "--steps", str(remaining_steps),
        "--sequences-per-block", str(plan.sequences_per_block), "--model-size", "substantive",
        "--architecture", "gdn2_hybrid", "--gdn-chunk-size",
        "32" if req.precision in {"fp16", "bf16"} else "64",
        "--initialization", "normal", "--optimizer", "hybrid_muon_adamw", "--device", "cuda",
        "--precision", req.precision, "--microbatch-size", str(req.microbatch_size),
        "--learning-rate", "3e-4", "--weight-decay", "0.1", "--muon-momentum", "0.95",
        "--muon-lr-multiplier", "1.0", "--muon-update-rms", "0.18", "--muon-weight-decay", "0.1",
        "--max-grad-norm", "1.0", "--schedule", "wsd", "--warmup-tokens", str(t["warmup_tokens"]),
        "--stable-tokens", str(t["stable_tokens"]), "--decay-tokens", str(t["decay_tokens"]),
        "--minimum-lr-ratio", str(t["minimum_lr_ratio"]), "--checkpoint-every-steps", "0",
        "--evaluation-every-steps", "0", "--validation-blocks", str(t["validation_blocks"]),
        "--checkpoint-at-steps", ",".join(map(str, spec.checkpoint_steps)),
        "--experiment-dir", str(spec.experiment_dir), "--probe-sequences", str(req.probe_sequences),
        "--source-commit", req.source_commit, "--seed", str(req.seed), "--wandb-mode", "disabled",
    ]
    if spec.profile_steps:
        command += ["--profile-at-steps", ",".join(map(str, spec.profile_steps))]
    if req.probe_lm_logits:
        command.append("--probe-lm-logits")
    if resume is not None:
        command += ["--resume", resume]
    if req.arm != "D":
        command += ["--load-balancing", "loss_free_sign" if req.arm == "M1" else "none",
                    "--balancing-step-size", format(0.0 if req.gamma is None else req.gamma, ".17g")]
    return command


def _event_checkpoint(spec: PilotSpec, event: Mapping[str, object]) -> tuple[str, Path]:
    checkpoint_id = event.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or _CHECKPOINT_ID.fullmatch(checkpoint_id) is None:
        raise RuntimeError("child local_checkpoint event has an invalid checkpoint ID")
    path = spec.checkpoint_dir / checkpoint_id
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"child announced a checkpoint that is not a local directory: {path}")
    try:
        complete = _complete_checkpoint(path)
        if complete["step"] > spec.request.steps:
            raise ValueError("checkpoint exceeds the frozen cap")
    except (OSError, EOFError, ValueError, TypeError, AttributeError, ImportError,
            IndexError, pickle.PickleError, RuntimeError) as error:
        raise RuntimeError(f"child announced an incomplete or invalid checkpoint: {path}") from error
    return checkpoint_id, path


def _runtime(path: Path) -> float:
    try:
        value = read_json(path).get("seconds")
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0
    except (OSError, ValueError, TypeError):
        return 0.0


def _result(spec: PilotSpec, runtime: float) -> dict[str, object]:
    latest = find_latest_complete_checkpoint(spec.checkpoint_dir)
    if latest is None:
        raise RuntimeError("pilot has no complete local joint checkpoint")
    if latest["step"] > spec.request.steps:
        raise RuntimeError("pilot checkpoint exceeds the frozen absolute step cap")
    return {"status": "complete" if latest["step"] >= spec.request.steps else "partial",
            "provider": spec.request.provider, "arm": spec.request.arm, "run_id": spec.request.run_id,
            "namespace": str(spec.namespace), "checkpoint_id": latest["checkpoint_id"],
            "completed_steps": latest["step"], "consumed_tokens": latest["consumed_tokens"],
            "total_runtime_seconds": runtime, "dataset_manifest_sha256": spec.plan.manifest_sha256,
            "actual_sdk_verified": False}


def run_pilot(request: PilotRequest, *, checkpoint_callback: Callable[[str, Path], None] | None = None,
              final_callback: Callable[[], None] | None = None,
              popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen) -> dict[str, object]:
    started = time.perf_counter()
    spec = prepare_pilot(request)
    latest = find_latest_complete_checkpoint(spec.checkpoint_dir)
    previous = 0 if latest is None else int(latest["step"])
    if previous > request.steps:
        raise RuntimeError("latest complete checkpoint exceeds the absolute step cap")
    remaining, runtime_path = request.steps - previous, spec.namespace / "runtime.json"
    accumulated = _runtime(runtime_path)
    try:
        if remaining:
            command = build_pilot_command(spec, remaining_steps=remaining,
                                          resume=None if latest is None else str(latest["checkpoint_id"]))
            logs = spec.namespace / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            log_path = logs / f"attempt-{len(tuple(logs.glob('attempt-*.log'))):04d}.log"
            env = os.environ.copy()
            env.update({"PYTHONUNBUFFERED": "1", "SMALL_LLM_EXPERIMENT_PROVIDER": request.provider,
                        "SMALL_LLM_EXPERIMENT_RUN_ID": request.run_id,
                        "SMALL_LLM_EXPERIMENT_ARM": request.arm,
                        "SMALL_LLM_EXPERIMENT_NAMESPACE": str(spec.namespace)})
            process = popen_factory(command, cwd=str(Path(__file__).resolve().parent), env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                if process.stdout is None:
                    raise RuntimeError("pilot child did not expose a stdout stream")
                with log_path.open("w", encoding="utf-8") as log:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
                        try:
                            value = json.loads(line)
                        except (ValueError, TypeError):
                            continue
                        event = value.get("local_checkpoint") if isinstance(value, Mapping) else None
                        if isinstance(event, Mapping) and checkpoint_callback is not None:
                            checkpoint_callback(*_event_checkpoint(spec, event))
                code = process.wait()
            except BaseException:
                process.kill()
                process.wait()
                raise
            if code != 0:
                raise RuntimeError(f"pilot child exited with status {code}")
        result = _result(spec, accumulated + time.perf_counter() - started)
        if result["completed_steps"] < request.steps:
            raise RuntimeError(f"pilot child stopped before its absolute cap: {result['completed_steps']} < {request.steps}")
        return result
    finally:
        write_json_atomic(runtime_path, {
            "seconds": accumulated + time.perf_counter() - started,
            "scope": "completed/failed attempts through result verification; excludes final volume commit, provider startup and abrupt termination",
        })
        if final_callback is not None:
            final_callback()


def add_pilot_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--arm", choices=PILOT_ARMS, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model", choices=(PILOT_MODEL,), default=PILOT_MODEL)
    parser.add_argument("--tokens", choices=(PILOT_TOKENS,), default=PILOT_TOKENS)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"), required=True)
    parser.add_argument("--microbatch-size", type=int, required=True)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--probe-sequences", type=int, default=2)
    parser.add_argument("--probe-lm-logits", action="store_true")
    parser.add_argument("--checkpoint-at-steps", type=_steps)
    parser.add_argument("--profile-at-steps", type=_steps)
    parser.add_argument("--source-commit")
    parser.add_argument("--dry-run", action="store_true")


def parse_pilot_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one bounded Small-LLM pilot arm")
    add_pilot_arguments(parser)
    args = parser.parse_args(argv)
    if args.arm == "M1" and args.gamma is None:
        parser.error("M1 requires an explicit --gamma")
    args.gamma = 0.0 if args.gamma is None else args.gamma
    if args.gamma < 0 or not math.isfinite(args.gamma):
        parser.error("--gamma must be finite and non-negative")
    if args.steps <= 0 or args.microbatch_size <= 0 or args.microbatch_size > 64:
        parser.error("steps must be positive and microbatch-size must be in 1..64")
    if args.probe_sequences < 0:
        parser.error("--probe-sequences cannot be negative")
    if args.probe_lm_logits and args.probe_sequences == 0:
        parser.error("--probe-lm-logits requires --probe-sequences")
    if args.arm != "M1" and args.gamma != 0:
        parser.error("only M1 accepts a non-zero --gamma")
    if args.source_commit is not None:
        try:
            _sha(args.source_commit)
        except ValueError as error:
            parser.error(str(error))
    try:
        _safe_leaf(args.run_id, "run_id")
    except ValueError as error:
        parser.error(str(error))
    return args


def request_from_args(args: argparse.Namespace, *, provider: str, run_root: Path,
                      source_commit: str) -> PilotRequest:
    if args.model != PILOT_MODEL or args.tokens != PILOT_TOKENS:
        raise ValueError("production pilots are frozen to the 100M/2B prepared dataset")
    return PilotRequest(provider, args.arm, args.run_id, Path(args.dataset_dir), run_root,
                        args.steps, args.precision, args.microbatch_size, _sha(source_commit),
                        args.seed, args.gamma, args.probe_sequences, args.probe_lm_logits,
                        args.checkpoint_at_steps, args.profile_at_steps)


def request_from_payload(payload: Mapping[str, object], *, provider: str, run_root: Path) -> PilotRequest:
    required = ("arm", "run_id", "dataset_dir", "steps", "precision", "microbatch_size",
                "source_commit", "seed", "probe_sequences")
    if any(name not in payload for name in required):
        raise ValueError("pilot payload is missing a required field")
    def text(name: str) -> str:
        value = payload[name]
        if not isinstance(value, str):
            raise ValueError(f"pilot payload {name} must be a string")
        return value
    def integer(name: str) -> int:
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"pilot payload {name} must be an integer")
        return value
    def optional_steps(name: str) -> tuple[int, ...] | None:
        value = payload.get(name)
        if value is None:
            return None
        if not isinstance(value, (tuple, list)) or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
            raise ValueError(f"pilot payload {name} must be a list of integers")
        return tuple(value)
    gamma = payload.get("gamma")
    if gamma is not None and (isinstance(gamma, bool) or not isinstance(gamma, (int, float))):
        raise ValueError("pilot payload gamma must be numeric or null")
    logits = payload.get("probe_lm_logits", False)
    if not isinstance(logits, bool):
        raise ValueError("pilot payload probe_lm_logits must be boolean")
    return PilotRequest(provider, text("arm"), text("run_id"), Path(text("dataset_dir")), run_root,
                        integer("steps"), text("precision"), integer("microbatch_size"),
                        _sha(text("source_commit")), integer("seed"),
                        None if gamma is None else float(gamma), integer("probe_sequences"), logits,
                        optional_steps("checkpoint_steps"), optional_steps("profile_steps"))


def request_payload(request: PilotRequest) -> dict[str, object]:
    validate_request(request)
    return {"arm": request.arm, "run_id": request.run_id, "dataset_dir": str(request.dataset_dir),
            "steps": request.steps, "precision": request.precision, "microbatch_size": request.microbatch_size,
            "source_commit": request.source_commit, "seed": request.seed, "gamma": request.gamma,
            "probe_sequences": request.probe_sequences, "probe_lm_logits": request.probe_lm_logits,
            "checkpoint_steps": request.checkpoint_steps, "profile_steps": request.profile_steps}


def prepare_provider_payload(
    payload: Mapping[str, object], *, provider: str, run_root: Path, dataset_root: Path,
    volume_commit: Callable[[], None],
) -> dict[str, object]:
    """CPU-side provider boundary: verify data, freeze identity, commit contract."""

    request = request_from_payload(payload, provider=provider, run_root=run_root)
    try:
        request.dataset_dir.resolve().relative_to(dataset_root.resolve())
    except ValueError as error:
        raise ValueError(f"prepared dataset must be under mounted {dataset_root}") from error
    spec = prepare_pilot(request)
    volume_commit()
    return {
        "status": "ready", "provider": provider, "arm": request.arm, "run_id": request.run_id,
        "namespace": str(spec.namespace), "dataset_manifest_sha256": spec.plan.manifest_sha256,
        "schedule": dict(spec.plan.trainer), "checkpoint_steps": list(spec.checkpoint_steps),
        "profile_steps": list(spec.profile_steps), "actual_sdk_verified": False,
    }


def run_provider_payload(
    payload: Mapping[str, object], *, provider: str, run_root: Path,
    volume_commit: Callable[[], None],
) -> dict[str, object]:
    """GPU-side provider boundary with checkpoint and final volume commits."""

    request = request_from_payload(payload, provider=provider, run_root=run_root)
    return run_pilot(
        request,
        checkpoint_callback=lambda _checkpoint_id, _checkpoint_path: volume_commit(),
        final_callback=volume_commit,
    )


def dry_run_payload(request: PilotRequest) -> dict[str, object]:
    validate_request(request)
    return {"status": "dry_run", "provider": request.provider, "arm": request.arm,
            "model": PILOT_MODEL, "tokens": PILOT_TOKENS, "run_id": request.run_id,
            "steps_cap": request.steps, "precision": request.precision,
            "microbatch_size": request.microbatch_size,
            "gamma": 0.0 if request.gamma is None else request.gamma,
            "dataset_dir": str(request.dataset_dir), "local_execution": False,
            "actual_sdk_verified": False}


__all__ = ["DEFAULT_CHECKPOINT_STEPS", "DEFAULT_PROFILE_STEPS", "PILOT_ARMS", "PILOT_MODEL",
           "PILOT_PROFILE", "PILOT_TOKENS", "PilotPlan", "PilotRequest", "PilotSpec",
           "add_pilot_arguments", "build_pilot_command", "dry_run_payload",
           "find_latest_complete_checkpoint", "parse_pilot_args", "prepare_pilot",
           "prepare_provider_payload", "request_from_args", "request_from_payload",
           "request_payload", "run_pilot", "run_provider_payload", "validate_request"]
