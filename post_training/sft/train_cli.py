"""Production SFT trainer over one verified immutable SFT bundle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import Mapping, Sequence

import torch

from dataset.src.joint_checkpoint import CheckpointCoordinator, verify_local_manifest
from dataset.src.remote import HuggingFaceCheckpointStore, TwoPhaseCheckpointPublisher
from trainer.engine import TrainerEngine, seed_everything
from trainer.session import TrainingSession

from .behavior_eval import evaluate_behavior
from .bundle import sft_budget_from_parent, verify_bundle
from .checkpoints import (
    download_parent_checkpoint,
    load_verified_native_checkpoint,
    sft_checkpoint_hashes,
)
from .config import SFTSchedulePlan, build_s0_trainer_config
from .publication import publication_dataset_manifest
from .storage import SFTShardReader

CHECKPOINT_EVERY = 250
EVALUATION_EVERY = 250
REMOTE_EVERY = 250
_CHECKPOINT_ID = re.compile(r"^step-(\d{8})$")


def _read_mapping(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError(f"{label} is missing or invalid: {path}") from error
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return dict(payload)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _checkpoint_step(checkpoint_id: str) -> int:
    match = _CHECKPOINT_ID.fullmatch(checkpoint_id)
    if match is None:
        raise RuntimeError(f"invalid SFT checkpoint ID: {checkpoint_id!r}")
    return int(match.group(1))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--sft-run-id", required=True)

    parent = parser.add_argument_group("parent checkpoint")
    parent.add_argument("--parent-checkpoint-dir", type=Path)
    parent.add_argument("--parent-repo-id")
    parent.add_argument("--parent-run-id")
    parent.add_argument("--parent-pointer", choices=("best", "latest"), default="best")
    parent.add_argument("--parent-revision")

    runtime = parser.add_argument_group("training")
    runtime.add_argument("--device", default="cuda")
    runtime.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    runtime.add_argument("--microbatch-size", type=_positive_int, default=4)
    runtime.add_argument("--learning-rate", type=float, default=3e-5)
    runtime.add_argument("--seed", type=_non_negative_int, default=17)
    runtime.add_argument("--checkpoint-every-steps", type=_positive_int, default=CHECKPOINT_EVERY)
    runtime.add_argument("--evaluation-every-steps", type=_positive_int, default=EVALUATION_EVERY)
    runtime.add_argument("--remote-publish-every-steps", type=_positive_int, default=REMOTE_EVERY)
    runtime.add_argument("--validation-blocks", type=_positive_int, default=16)
    runtime.add_argument("--behavior-cases", type=_positive_int, default=16)
    runtime.add_argument("--max-steps-this-session", type=_positive_int)

    remote = parser.add_argument_group("remote SFT checkpoints")
    remote.add_argument("--checkpoint-repo-id")
    remote.add_argument("--checkpoint-revision")
    remote.add_argument("--token-env", default="HF_TOKEN")
    remote.add_argument("--create-checkpoint-repo", action="store_true")
    remote.add_argument(
        "--no-automatic-resume",
        action="store_true",
        help="disable verified local/remote automatic resume",
    )

    wandb = parser.add_argument_group("Weights & Biases")
    wandb.add_argument("--wandb-mode", choices=("disabled", "offline", "online"), default="online")
    wandb.add_argument("--wandb-project", default="Small-LLM")
    wandb.add_argument("--wandb-entity")
    wandb.add_argument("--wandb-run-id")
    wandb.add_argument("--wandb-run-name")
    wandb.add_argument("--wandb-dir", type=Path)
    return parser


def _resolve_parent(args: argparse.Namespace, token: str | None) -> tuple[Path, dict[str, object]]:
    if args.parent_checkpoint_dir is not None:
        root = args.parent_checkpoint_dir.resolve()
        return root, {"transport": "local", "path": str(root)}
    if not args.parent_repo_id or not args.parent_run_id:
        raise SystemExit(
            "pass --parent-checkpoint-dir or both --parent-repo-id and --parent-run-id"
        )
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix=".parent-", dir=args.checkpoint_dir))
    root, remote = download_parent_checkpoint(
        repo_id=args.parent_repo_id,
        run_id=args.parent_run_id,
        pointer=args.parent_pointer,
        token=token,
        revision=args.parent_revision,
        destination=destination,
    )
    return root, {"transport": "remote", **remote}


def _validate_resume_checkpoint(
    root: Path,
    *,
    expected_hashes: tuple[str, str, str],
    expected_identity: Mapping[str, object],
) -> tuple[str, int]:
    verify_local_manifest(root)
    payload = _read_mapping(root / "checkpoint.json", label="SFT resume checkpoint")
    checkpoint_id = payload.get("checkpoint_id")
    if not isinstance(checkpoint_id, str):
        raise RuntimeError("SFT resume checkpoint has no checkpoint_id")
    step = _checkpoint_step(checkpoint_id)
    if root.name != checkpoint_id:
        raise RuntimeError("SFT checkpoint directory name disagrees with checkpoint_id")
    observed_hashes = (
        payload.get("configuration_hash"),
        payload.get("source_hash"),
        payload.get("schema_hash"),
    )
    if observed_hashes != expected_hashes:
        raise RuntimeError(f"SFT resume checkpoint identity mismatch: {checkpoint_id}")
    pipeline = payload.get("pipeline_state")
    if not isinstance(pipeline, Mapping) or pipeline.get("sft_identity") != expected_identity:
        raise RuntimeError(f"SFT resume pipeline identity mismatch: {checkpoint_id}")
    return checkpoint_id, step


def _local_resume(
    args: argparse.Namespace,
    *,
    expected_hashes: tuple[str, str, str],
    expected_identity: Mapping[str, object],
) -> tuple[Path, dict[str, object]] | None:
    if args.no_automatic_resume or not args.checkpoint_dir.is_dir():
        return None
    candidates: list[tuple[int, Path, str]] = []
    for path in sorted(args.checkpoint_dir.iterdir()):
        if not path.is_dir() or _CHECKPOINT_ID.fullmatch(path.name) is None:
            continue
        checkpoint_id, step = _validate_resume_checkpoint(
            path,
            expected_hashes=expected_hashes,
            expected_identity=expected_identity,
        )
        candidates.append((step, path, checkpoint_id))
    if not candidates:
        return None
    step, path, checkpoint_id = max(candidates, key=lambda item: item[0])
    return path, {
        "transport": "local",
        "checkpoint_id": checkpoint_id,
        "step": step,
        "path": str(path),
    }


def _remote_resume(
    args: argparse.Namespace,
    *,
    token: str | None,
    expected_hashes: tuple[str, str, str],
    expected_identity: Mapping[str, object],
) -> tuple[Path, dict[str, object]] | None:
    if args.no_automatic_resume or not args.checkpoint_repo_id:
        return None
    store = HuggingFaceCheckpointStore(
        args.checkpoint_repo_id,
        token=token,
        private=True,
        revision=args.checkpoint_revision,
    )
    pointer = store.read_json(f"run/{args.sft_run_id}/latest.json")
    if pointer is None:
        return None
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix=".restore-", dir=args.checkpoint_dir))
    from trainer.post_pretraining_prompt_suite import download_verified_checkpoint

    root, remote = download_verified_checkpoint(
        repo_id=args.checkpoint_repo_id,
        run_id=args.sft_run_id,
        token=token,
        revision=args.checkpoint_revision,
        pointer_name="latest",
        destination=destination,
    )
    checkpoint_id, step = _validate_resume_checkpoint(
        root,
        expected_hashes=expected_hashes,
        expected_identity=expected_identity,
    )
    return root, {"transport": "remote", **dict(remote), "checkpoint_id": checkpoint_id, "step": step}


def _select_resume(
    args: argparse.Namespace,
    *,
    token: str | None,
    expected_hashes: tuple[str, str, str],
    expected_identity: Mapping[str, object],
) -> tuple[Path, dict[str, object]] | None:
    local = _local_resume(
        args,
        expected_hashes=expected_hashes,
        expected_identity=expected_identity,
    )
    remote = _remote_resume(
        args,
        token=token,
        expected_hashes=expected_hashes,
        expected_identity=expected_identity,
    )
    if local is None:
        return remote
    if remote is None:
        return local
    local_step = int(local[1]["step"])
    remote_step = int(remote[1]["step"])
    return remote if remote_step > local_step else local


def _wandb_run(
    args: argparse.Namespace,
    *,
    parent_identity: Mapping[str, object],
    bundle_manifest: Mapping[str, object],
    trainer_config: Mapping[str, object],
    resumed: bool,
):
    if args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("SFT W&B telemetry requires wandb") from error
    directory = args.wandb_dir or (args.checkpoint_dir / "wandb")
    directory.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, object] = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "name": args.wandb_run_name,
        "mode": args.wandb_mode,
        "dir": str(directory),
        "job_type": "sft-qualification",
        "config": {
            "sft_run_id": args.sft_run_id,
            "parent": dict(parent_identity),
            "bundle_manifest_sha256": bundle_manifest["manifest_sha256"],
            "trainer": dict(trainer_config),
        },
        "tags": ["sft", "20m", "microbatch-4", "one-pass", "segmented-exact-resume"],
    }
    if args.wandb_run_id:
        kwargs["id"] = args.wandb_run_id
        kwargs["resume"] = "must" if resumed else "allow"
    run = wandb.init(**kwargs)
    if run is None:
        raise RuntimeError("wandb.init did not return a run")
    run.define_metric("trainer/global_step")
    run.define_metric("*", step_metric="trainer/global_step")
    return run


def _log(run: object | None, payload: Mapping[str, object]) -> None:
    if run is not None:
        run.log(dict(payload))


def _memory_snapshot(engine: TrainerEngine) -> dict[str, int]:
    """Return lightweight host/GPU memory telemetry without adding dependencies."""

    result: dict[str, int] = {}
    status = Path("/proc/self/status")
    if status.is_file():
        try:
            for line in status.read_text(encoding="utf-8").splitlines():
                key, separator, value = line.partition(":")
                if not separator or key not in {"VmRSS", "VmHWM"}:
                    continue
                fields = value.strip().split()
                if not fields:
                    continue
                kib = int(fields[0])
                result["host_rss_bytes" if key == "VmRSS" else "host_peak_rss_bytes"] = kib * 1024
        except (OSError, ValueError):
            pass
    if engine.device.type == "cuda" and torch.cuda.is_available():
        result.update(
            cuda_allocated_bytes=int(torch.cuda.memory_allocated(engine.device)),
            cuda_reserved_bytes=int(torch.cuda.memory_reserved(engine.device)),
            cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(engine.device)),
            cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(engine.device)),
        )
    return result


def _cadence_actions(
    step: int,
    *,
    checkpoint_every_steps: int,
    remote_publish_every_steps: int,
    evaluation_every_steps: int,
) -> tuple[str, ...]:
    """Order durability before evaluation at every cadence boundary."""

    actions: list[str] = []
    checkpoint_due = step % checkpoint_every_steps == 0
    publish_due = step % remote_publish_every_steps == 0
    evaluation_due = step % evaluation_every_steps == 0
    if checkpoint_due or publish_due:
        actions.append("checkpoint")
    if publish_due:
        actions.append("publish")
    if evaluation_due:
        actions.append("evaluate")
    return tuple(actions)


def _validation(
    engine: TrainerEngine,
    bundle_root: Path,
    *,
    maximum_blocks: int,
) -> dict[str, object]:
    reader = SFTShardReader(bundle_root / "validation", split="validation")
    started = time.perf_counter()
    result = engine.evaluate(reader.iter_from_start(), maximum_batches=maximum_blocks)
    return {**result, "elapsed_seconds": time.perf_counter() - started}


def _checkpoint_pipeline_identity(
    *,
    parent_identity: Mapping[str, object],
    bundle_manifest: Mapping[str, object],
) -> dict[str, object]:
    return {
        "stage": "sft_s0",
        "parent_checkpoint_identity": parent_identity["identity_sha256"],
        "bundle_manifest_identity": bundle_manifest["manifest_sha256"],
        "template_identity": "small-llm-s0-v1",
        "loss_identity": "assistant-only-ce-v1",
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seed_everything(args.seed)

    bundle_root = args.dataset_dir.resolve()
    verification = verify_bundle(bundle_root)
    bundle_manifest = _read_mapping(bundle_root / "bundle-manifest.json", label="SFT bundle manifest")
    train_reader = SFTShardReader(bundle_root / "train", split="train")
    schedule = SFTSchedulePlan.from_block_target_counts(train_reader.block_target_counts)
    trainer_config = build_s0_trainer_config(
        schedule,
        microbatch_size=args.microbatch_size,
        precision=args.precision,
        seed=args.seed,
        learning_rate=args.learning_rate,
        checkpoint_every_steps=args.checkpoint_every_steps,
        evaluation_every_steps=args.evaluation_every_steps,
    )

    token = os.environ.get(args.token_env)
    parent_root, parent_transport = _resolve_parent(args, token)
    parent_model, model_config, parent_identity = load_verified_native_checkpoint(parent_root, device="cpu")
    parent_identity = {**parent_identity, "transport": parent_transport}

    parent_consumed = parent_identity.get("consumed_tokens")
    if isinstance(parent_consumed, bool) or not isinstance(parent_consumed, int):
        raise RuntimeError("verified parent has no integer consumed-token count")
    expected_sft_targets = sft_budget_from_parent(parent_consumed)
    requested_sft_targets = bundle_manifest.get("train_target_tokens_requested")
    if requested_sft_targets != expected_sft_targets:
        raise RuntimeError(
            "SFT bundle does not implement the frozen 4% budget for this parent: "
            f"parent={parent_consumed}, expected={expected_sft_targets}, "
            f"bundle_requested={requested_sft_targets}"
        )

    hashes = sft_checkpoint_hashes(
        parent_identity=parent_identity,
        bundle_manifest=bundle_manifest,
        trainer_config=trainer_config.as_dict(),
    )
    coordinator = CheckpointCoordinator(
        args.checkpoint_dir,
        configuration_hash=hashes[0],
        source_hash=hashes[1],
        schema_hash=hashes[2],
    )
    engine = TrainerEngine(parent_model, trainer_config, device=args.device)
    session = TrainingSession(engine, train_reader)
    identity = _checkpoint_pipeline_identity(
        parent_identity=parent_identity,
        bundle_manifest=bundle_manifest,
    )

    resumed = _select_resume(
        args,
        token=token,
        expected_hashes=hashes,
        expected_identity=identity,
    )
    resume_info: dict[str, object] | None = None
    remotely_present_checkpoint: str | None = None
    if resumed is not None:
        resume_root, resume_info = resumed
        checkpoint_id = str(resume_info["checkpoint_id"])
        target = args.checkpoint_dir / checkpoint_id
        if resume_info.get("transport") == "remote":
            remotely_present_checkpoint = checkpoint_id
            if not target.exists():
                shutil.copytree(resume_root, target)
        pipeline = session.load_checkpoint(coordinator, checkpoint_id)
        if pipeline.get("sft_identity") != identity:
            raise RuntimeError("SFT resume identity does not match parent/data/objective")
        print(
            json.dumps(
                {
                    "resume": {
                        "transport": resume_info.get("transport"),
                        "checkpoint_id": checkpoint_id,
                        "global_step": engine.global_step,
                        "consumed_targets": engine.consumed_tokens,
                    }
                },
                sort_keys=True,
            ),
            flush=True,
        )

    publisher = None
    publication_manifest = None
    if args.checkpoint_repo_id:
        store = HuggingFaceCheckpointStore(
            args.checkpoint_repo_id,
            token=token,
            private=True,
            revision=args.checkpoint_revision,
            create_repo=args.create_checkpoint_repo,
        )
        publisher = TwoPhaseCheckpointPublisher(store, run_id=args.sft_run_id)
        publication_manifest = publication_dataset_manifest(bundle_root, run_id=args.sft_run_id)

    run = _wandb_run(
        args,
        parent_identity=parent_identity,
        bundle_manifest=bundle_manifest,
        trainer_config=trainer_config.as_dict(),
        resumed=resumed is not None,
    )
    saved: set[str] = set()
    published: set[str] = set()
    if resumed is not None:
        saved.add(str(resume_info["checkpoint_id"]))
    if remotely_present_checkpoint is not None:
        published.add(remotely_present_checkpoint)
    validation: dict[str, object] | None = None
    behavior: dict[str, object] | None = None

    def cadence_event(phase: str, **extra: object) -> None:
        event: dict[str, object] = {
            "step": engine.global_step,
            "phase": phase,
            "memory": _memory_snapshot(engine),
            **extra,
        }
        print(json.dumps({"sft_cadence": event}, sort_keys=True), flush=True)

    def evaluate(*, full_behavior: bool = False) -> None:
        nonlocal validation, behavior
        cadence_event("validation:start", maximum_blocks=args.validation_blocks)
        validation = _validation(engine, bundle_root, maximum_blocks=args.validation_blocks)
        cadence_event("validation:done", **validation)
        selected_behavior_cases = None if full_behavior else args.behavior_cases
        cadence_event("behavior:start", maximum_cases=selected_behavior_cases)
        behavior = evaluate_behavior(
            engine.model,
            precision=args.precision,
            max_seq_len=model_config.max_seq_len,
            max_cases=selected_behavior_cases,
        )
        cadence_event("behavior:done", **behavior["summary"])
        event = {"step": engine.global_step, "validation": validation, "behavior": behavior["summary"]}
        print(json.dumps({"sft_evaluation": event}, sort_keys=True), flush=True)
        _log(
            run,
            {
                "trainer/global_step": engine.global_step,
                "validation/loss": validation["loss"],
                "validation/perplexity": validation["perplexity"],
                "sft_behavior/pass_rate": behavior["summary"]["pass_rate"],
                "sft_behavior/eos_termination_rate": behavior["summary"]["eos_termination_rate"],
                "sft_behavior/runaway_rate": behavior["summary"]["runaway_rate"],
            },
        )

    def save(checkpoint_id: str) -> None:
        if checkpoint_id in saved:
            return
        cadence_event("checkpoint:start", checkpoint_id=checkpoint_id)
        pipeline = {**train_reader.pipeline_state(), "sft_identity": identity}
        metrics = {
            "sft_validation": validation or {},
            "sft_behavior": behavior["summary"] if behavior is not None else {},
        }
        session.save_checkpoint(
            coordinator,
            checkpoint_id,
            pipeline_state=pipeline,
            validation_metrics=metrics,
        )
        saved.add(checkpoint_id)
        cadence_event("checkpoint:done", checkpoint_id=checkpoint_id)
        print(json.dumps({"local_checkpoint": checkpoint_id}, sort_keys=True), flush=True)

    def publish(checkpoint_id: str, *, final: bool) -> None:
        if publisher is None or publication_manifest is None or checkpoint_id in published:
            return
        save(checkpoint_id)
        cadence_event("publication:start", checkpoint_id=checkpoint_id, final=final)
        coordinator.publish(
            publisher,
            checkpoint_id=checkpoint_id,
            drive_manifest=publication_manifest,
            metric=None,
            best_metric=None,
        )
        published.add(checkpoint_id)
        cadence_event("publication:done", checkpoint_id=checkpoint_id, final=final)
        print(
            json.dumps({"remote_publication": {"checkpoint_id": checkpoint_id, "final": final}}, sort_keys=True),
            flush=True,
        )

    total_steps = train_reader.block_count
    if engine.global_step > total_steps:
        raise RuntimeError("SFT checkpoint is beyond the verified training stream")
    remaining = total_steps - engine.global_step
    session_limit = remaining if args.max_steps_this_session is None else min(
        remaining, args.max_steps_this_session
    )

    try:
        for _ in range(session_limit):
            metrics = session.step()
            print(json.dumps(metrics.as_dict(), sort_keys=True), flush=True)
            _log(
                run,
                {
                    "trainer/global_step": engine.global_step,
                    **{f"train/{key}": value for key, value in metrics.as_dict().items() if key != "step"},
                },
            )
            checkpoint_id = f"step-{engine.global_step:08d}"
            for action in _cadence_actions(
                engine.global_step,
                checkpoint_every_steps=args.checkpoint_every_steps,
                remote_publish_every_steps=args.remote_publish_every_steps,
                evaluation_every_steps=args.evaluation_every_steps,
            ):
                if action == "checkpoint":
                    save(checkpoint_id)
                elif action == "publish":
                    publish(checkpoint_id, final=False)
                elif action == "evaluate":
                    evaluate()
                else:  # pragma: no cover - protected by _cadence_actions
                    raise RuntimeError(f"unknown SFT cadence action: {action}")

        checkpoint_id = f"step-{engine.global_step:08d}"
        save(checkpoint_id)
        publish(checkpoint_id, final=engine.global_step == total_steps)
        evaluate(full_behavior=engine.global_step == total_steps)
        summary = {
            "schema": "small-llm-sft-training-summary-v1",
            "sft_run_id": args.sft_run_id,
            "parent": parent_identity,
            "bundle": verification,
            "budget": {
                "parent_consumed_tokens": parent_consumed,
                "fraction": 0.04,
                "requested_loss_bearing_target_tokens": expected_sft_targets,
            },
            "global_step": engine.global_step,
            "total_steps": total_steps,
            "consumed_loss_bearing_target_tokens": engine.consumed_tokens,
            "complete": engine.global_step == total_steps,
            "checkpoint_id": checkpoint_id,
            "validation": validation,
            "behavior": behavior["summary"] if behavior else None,
            "resume": resume_info,
        }
        summary_path = args.checkpoint_dir / "sft-summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"sft_summary": summary}, sort_keys=True), flush=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except BaseException:
        if run is not None:
            try:
                run.finish(exit_code=1)
            except Exception:
                pass
        raise

    if run is not None:
        run.finish(exit_code=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "_cadence_actions",
    "_checkpoint_step",
    "_memory_snapshot",
    "_select_resume",
    "_validate_resume_checkpoint",
    "build_parser",
    "main",
]
