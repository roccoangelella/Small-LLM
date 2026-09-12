"""Provider-neutral execution contract for the accepted production MoE.

This module deliberately does not reuse ``moe_pilot``: the pilot freezes the historical
D/M0/M1 100M/2B experiment identities, while production must always enter
``MoEModelConfig.accepted()`` (64 experts, Top-2, width-352 experts, Quantile Balancing,
8,000 semantic token IDs with 8,192-row physical embedding storage).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
import re
from pathlib import Path
import subprocess
import sys
from typing import Callable

from MOE_model.config import MoEModelConfig
from dataset.incremental_frontier import (
    DEFAULT_TRAINING_VALIDATION_BLOCKS,
    RUN_CONTRACT_FILENAME,
    standard_wsd_plan,
)
from dataset.src.checkpoint_sequence import (
    CHECKPOINT_ID, complete_checkpoint, find_latest_complete_checkpoint,
)
from trainer.shards import SchemaV2ShardReader


_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_SHA = re.compile(r"[0-9a-f]{40}")

RESUME_LATEST = "latest"
_SCHEDULE_KEYS = ("steps", "warmup_tokens", "stable_tokens", "decay_tokens", "minimum_lr_ratio")
# The recipe the pilot pins explicitly (moe_pilot.build_pilot_command); the trainer defaults
# reproduce them today, but a production command must not depend on defaults.
# Placeholder for launcher dry-run displays, where the dataset volume is not mounted;
# the real values are read from the run contract inside the container.
DISPLAY_SCHEDULE = {
    "steps": 10**12, "warmup_tokens": 0, "stable_tokens": 0, "decay_tokens": 0,
    "minimum_lr_ratio": 0.1, "validation_blocks": DEFAULT_TRAINING_VALIDATION_BLOCKS,
}
OPTIMIZER_RECIPE = (
    "--learning-rate", "3e-4", "--weight-decay", "0.1", "--muon-momentum", "0.95",
    "--muon-lr-multiplier", "1.0", "--muon-update-rms", "0.18", "--muon-weight-decay", "0.1",
    "--max-grad-norm", "1.0",
)


@dataclass(frozen=True, slots=True)
class ProductionRequest:
    """One production run whose ``total_steps`` is the absolute update target.

    The trainer consumes ``--steps`` as a per-segment count, so a resumed segment
    is launched with the remaining updates rather than the absolute target.
    """

    run_id: str
    dataset_dir: str
    total_steps: int
    precision: str
    microbatch_size: int
    source_commit: str
    resume: str | None = None
    sequences_per_block: int | None = None
    checkpoint_every_steps: int = 1000
    validation_blocks: int | None = None
    keep_last_checkpoints: int = 3
    milestone_every_steps: int = 0
    max_wall_seconds: float = 0.0
    compile_mode: str = "off"

    def __post_init__(self) -> None:
        if _RUN_ID.fullmatch(self.run_id) is None:
            raise ValueError("run_id must contain only letters, digits, '.', '_' or '-'")
        if not self.dataset_dir:
            raise ValueError("dataset_dir is required")
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if self.precision not in {"fp16", "bf16", "fp32"}:
            raise ValueError("precision must be fp16, bf16 or fp32")
        if self.microbatch_size <= 0:
            raise ValueError("microbatch_size must be positive")
        if _SHA.fullmatch(self.source_commit) is None:
            raise ValueError("source_commit must be a full lowercase Git SHA")
        if self.sequences_per_block is not None and self.sequences_per_block <= 0:
            raise ValueError("sequences_per_block must be positive when supplied")
        if self.checkpoint_every_steps < 0:
            raise ValueError("checkpoint_every_steps cannot be negative")
        if self.validation_blocks is not None and self.validation_blocks < 0:
            raise ValueError("validation_blocks cannot be negative")
        if self.keep_last_checkpoints < 0 or self.keep_last_checkpoints == 1:
            raise ValueError("keep_last_checkpoints must be 0 or at least 2")
        if self.milestone_every_steps < 0:
            raise ValueError("milestone_every_steps cannot be negative")
        if self.max_wall_seconds < 0:
            raise ValueError("max_wall_seconds cannot be negative")
        if self.compile_mode not in {"off", "blocks"}:
            raise ValueError("compile_mode must be off or blocks")
        if (
            self.resume not in (None, "", RESUME_LATEST)
            and CHECKPOINT_ID.fullmatch(self.resume) is None
        ):
            raise ValueError("resume must be a step checkpoint ID or the literal 'latest'")


def accepted_identity() -> dict[str, object]:
    config = MoEModelConfig.accepted()
    return {
        "model_size": "accepted",
        "num_experts": config.num_experts,
        "top_k": config.top_k,
        "expert_d_ff": config.expert_d_ff,
        "router_scoring": config.router_scoring,
        "load_balancing": config.load_balancing,
        "semantic_vocab_size": config.semantic_vocab_size,
        "padded_vocab_size": config.padded_vocab_size,
        "gdn_chunk_size": config.dense.gdn_chunk_size,
        "dispatch": config.dispatch,
    }


def request_payload(request: ProductionRequest) -> dict[str, object]:
    return {"request": asdict(request), "identity": accepted_identity()}


def request_from_payload(payload: dict[str, object]) -> ProductionRequest:
    request = payload.get("request")
    if not isinstance(request, dict):
        raise ValueError("production payload has no request object")
    return ProductionRequest(**request)  # type: ignore[arg-type]


def resolve_schedule(dataset_dir: str | Path) -> dict[str, object]:
    """Read the WSD schedule the corpus was produced for, from its run contract.

    Rocco's producer writes the full ``trainer`` plan into ``run_contract.json``; a
    retokenized finite corpus carries only the block geometry, from which the same
    ``standard_wsd_plan`` (5 % warmup, 75 % stable, 20 % decay, floor 0.1) is derived.
    """

    path = Path(dataset_dir) / RUN_CONTRACT_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"production dataset has no {RUN_CONTRACT_FILENAME}: {path}")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(contract, Mapping):
        raise ValueError("run contract must be a JSON object")
    trainer = contract.get("trainer")
    if trainer is None:
        blocks = contract.get("planned_train_blocks")
        if isinstance(blocks, bool) or not isinstance(blocks, int) or blocks <= 0:
            raise ValueError("run contract has neither a trainer plan nor planned_train_blocks")
        trainer = standard_wsd_plan(
            blocks,
            context_length=int(contract["context_length"]),
            sequences_per_block=int(contract["sequences_per_block"]),
            validation_blocks=int(contract.get("validation_blocks", DEFAULT_TRAINING_VALIDATION_BLOCKS)),
        )
    if not isinstance(trainer, Mapping) or trainer.get("schedule") != "wsd":
        raise ValueError("run contract trainer plan must be a WSD schedule")
    missing = [key for key in _SCHEDULE_KEYS if key not in trainer]
    if missing:
        raise ValueError(f"run contract trainer plan is missing {missing}")
    schedule = {key: trainer[key] for key in (*_SCHEDULE_KEYS, "validation_blocks") if key in trainer}
    schedule.setdefault("validation_blocks", DEFAULT_TRAINING_VALIDATION_BLOCKS)
    return schedule


def checkpoint_dir(request: ProductionRequest, *, run_root: Path) -> Path:
    return run_root / request.run_id / "checkpoints"


def resolve_resume(request: ProductionRequest, *, run_root: Path) -> dict[str, object]:
    """Resolve the segment this attempt must run, so a retry continues the run.

    An absent or ``latest`` resume takes the newest complete local checkpoint;
    an explicit checkpoint ID is honoured as supplied.
    """

    if request.resume and request.resume != RESUME_LATEST:
        checkpoint = complete_checkpoint(checkpoint_dir(request, run_root=run_root) / request.resume)
        resume, completed = request.resume, int(checkpoint["step"])
    else:
        latest = find_latest_complete_checkpoint(checkpoint_dir(request, run_root=run_root))
        resume = None if latest is None else str(latest["checkpoint_id"])
        completed = 0 if latest is None else int(latest["step"])
    if completed > request.total_steps:
        raise RuntimeError("resumed checkpoint exceeds the absolute production step target")
    return {
        "resume": resume,
        "completed_steps": completed,
        "remaining_steps": request.total_steps - completed,
    }


def build_training_command(
    request: ProductionRequest, *, run_root: Path, plan: Mapping[str, object] | None = None,
    schedule: Mapping[str, object] | None = None,
) -> list[str]:
    """Build a command whose architecture identity cannot fall back to an M0/M1 pilot.

    The learning-rate schedule always comes from the corpus run contract (``schedule``
    or ``resolve_schedule(request.dataset_dir)``), never from trainer defaults.
    """

    schedule = schedule if schedule is not None else resolve_schedule(request.dataset_dir)
    if request.total_steps > int(schedule["steps"]):
        raise ValueError(
            f"total_steps {request.total_steps} exceeds the corpus plan of {schedule['steps']} updates"
        )
    validation_blocks = (
        int(schedule["validation_blocks"]) if request.validation_blocks is None
        else request.validation_blocks
    )
    plan = plan if plan is not None else resolve_resume(request, run_root=run_root)
    remaining = int(plan["remaining_steps"])
    if remaining <= 0:
        raise ValueError("the absolute production step target is already reached")
    resume = plan["resume"]
    run_dir = run_root / request.run_id
    command = [
        sys.executable,
        "-m",
        "MOE_model",
        "--dataset-dir",
        request.dataset_dir,
        "--checkpoint-dir",
        str(run_dir / "checkpoints"),
        "--experiment-dir",
        str(run_dir / "artifacts"),
        "--steps",
        str(remaining),
        "--model-size",
        "accepted",
        "--architecture",
        "gdn2_hybrid",
        "--gdn-chunk-size",
        "32",
        "--load-balancing",
        "quantile",
        "--balancing-step-size",
        "0",
        "--initialization",
        "normal",
        "--optimizer",
        "hybrid_muon_adamw",
        "--precision",
        request.precision,
        "--microbatch-size",
        str(request.microbatch_size),
        "--source-commit",
        request.source_commit,
        "--checkpoint-every-steps",
        str(request.checkpoint_every_steps),
        "--keep-last-checkpoints",
        str(request.keep_last_checkpoints),
        "--milestone-every-steps",
        str(request.milestone_every_steps),
        "--max-wall-seconds",
        format(float(request.max_wall_seconds), ".17g"),
        "--validation-blocks",
        str(validation_blocks),
        "--evaluation-every-steps",
        str(request.checkpoint_every_steps),
        "--compile",
        request.compile_mode,
        *OPTIMIZER_RECIPE,
        "--schedule",
        "wsd",
        "--warmup-tokens",
        str(int(schedule["warmup_tokens"])),
        "--stable-tokens",
        str(int(schedule["stable_tokens"])),
        "--decay-tokens",
        str(int(schedule["decay_tokens"])),
        "--minimum-lr-ratio",
        format(float(schedule["minimum_lr_ratio"]), ".17g"),
    ]
    if resume:
        command += ["--resume", str(resume)]
    if request.sequences_per_block is not None:
        command += ["--sequences-per-block", str(request.sequences_per_block)]
    return command


def prepare_dataset(request: ProductionRequest) -> dict[str, object]:
    """CPU gate the manifest geometry and decode the first train block with vocab=8000."""

    config = MoEModelConfig.accepted()
    root = Path(request.dataset_dir)
    reader = SchemaV2ShardReader(
        root,
        split="train",
        sequences_per_block=request.sequences_per_block,
        semantic_vocab_size=config.semantic_vocab_size,
        context_length=config.max_seq_len,
    )
    first = reader.next_batch()
    if first.sequence_count <= 0:
        raise RuntimeError("production dataset contains an empty first training block")
    return {
        "status": "ready",
        "dataset_dir": str(root),
        "train_blocks": reader.block_count,
        "first_block_id": first.block_id,
        "first_block_sequences": first.sequence_count,
        "identity": accepted_identity(),
    }


def _child_event(line: str, name: str) -> Mapping[str, object] | None:
    try:
        value = json.loads(line)
    except (ValueError, TypeError):
        return None
    if not isinstance(value, Mapping):
        return None
    event = value.get(name)
    return event if isinstance(event, Mapping) else None


def run_provider_payload(
    payload: dict[str, object],
    *,
    run_root: Path,
    repo_root: Path,
    volume_commit: Callable[[], object] | None = None,
    popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
) -> dict[str, object]:
    """Run one segment, committing the run volume behind every local checkpoint."""

    request = request_from_payload(payload)
    plan = resolve_resume(request, run_root=run_root)
    schedule = resolve_schedule(request.dataset_dir)
    result: dict[str, object] = {
        "status": "complete",
        "schedule": dict(schedule),
        "run_id": request.run_id,
        "steps_requested": request.total_steps,
        "total_steps": request.total_steps,
        "resumed_from": plan["resume"],
        "resumed_step": plan["completed_steps"],
        "steps_launched": plan["remaining_steps"],
        "identity": accepted_identity(),
        "checkpoint_dir": str(checkpoint_dir(request, run_root=run_root)),
    }
    if int(plan["remaining_steps"]) <= 0:
        result["steps_launched"] = 0
        if volume_commit is not None:
            volume_commit()
        return result

    command = build_training_command(request, run_root=run_root, plan=plan, schedule=schedule)
    committed: list[str] = []
    drained: Mapping[str, object] | None = None
    process = popen_factory(command, cwd=str(repo_root), text=True, bufsize=1,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        if process.stdout is None:
            raise RuntimeError("accepted MoE trainer did not expose a stdout stream")
        for line in process.stdout:
            print(line, end="", flush=True)
            checkpoint = _child_event(line, "local_checkpoint")
            if checkpoint is not None:
                checkpoint_id = checkpoint.get("checkpoint_id")
                if not isinstance(checkpoint_id, str) or CHECKPOINT_ID.fullmatch(checkpoint_id) is None:
                    raise RuntimeError("child local_checkpoint event has an invalid checkpoint ID")
                if volume_commit is not None:
                    volume_commit()
                committed.append(checkpoint_id)
                continue
            drained = _child_event(line, "drained") or drained
        code = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        raise
    if code != 0:
        raise RuntimeError(f"accepted MoE trainer exited with status {code}")
    if volume_commit is not None:
        volume_commit()
    result["committed_checkpoints"] = committed
    result["drained"] = None if drained is None else dict(drained)
    if drained is not None:
        result["status"] = "drained"
    return result


__all__ = [
    "RESUME_LATEST",
    "ProductionRequest",
    "accepted_identity",
    "build_training_command",
    "checkpoint_dir",
    "prepare_dataset",
    "request_from_payload",
    "request_payload",
    "resolve_resume",
    "run_provider_payload",
]
