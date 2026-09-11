"""Provider-neutral execution contract for the accepted production MoE.

This module deliberately does not reuse ``moe_pilot``: the pilot freezes the historical
D/M0/M1 100M/2B experiment identities, while production must always enter
``MoEModelConfig.accepted()`` (64 experts, Top-2, width-352 experts, Quantile Balancing,
8,000 semantic token IDs with 8,192-row physical embedding storage).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from pathlib import Path
import subprocess
import sys
from typing import Callable

from MOE_model.config import MoEModelConfig
from trainer.shards import SchemaV2ShardReader


_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_SHA = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True, slots=True)
class ProductionRequest:
    run_id: str
    dataset_dir: str
    steps: int
    precision: str
    microbatch_size: int
    source_commit: str
    resume: str | None = None
    sequences_per_block: int | None = None
    checkpoint_every_steps: int = 0
    validation_blocks: int = 0

    def __post_init__(self) -> None:
        if _RUN_ID.fullmatch(self.run_id) is None:
            raise ValueError("run_id must contain only letters, digits, '.', '_' or '-'")
        if not self.dataset_dir:
            raise ValueError("dataset_dir is required")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
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
        if self.validation_blocks < 0:
            raise ValueError("validation_blocks cannot be negative")


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


def build_training_command(request: ProductionRequest, *, run_root: Path) -> list[str]:
    """Build a command whose architecture identity cannot fall back to an M0/M1 pilot."""

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
        str(request.steps),
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
        "--validation-blocks",
        str(request.validation_blocks),
    ]
    if request.resume:
        command += ["--resume", request.resume]
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


def run_provider_payload(
    payload: dict[str, object],
    *,
    run_root: Path,
    repo_root: Path,
    volume_commit: Callable[[], object] | None = None,
) -> dict[str, object]:
    request = request_from_payload(payload)
    command = build_training_command(request, run_root=run_root)
    completed = subprocess.run(command, cwd=repo_root, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"accepted MoE trainer exited with status {completed.returncode}")
    if volume_commit is not None:
        volume_commit()
    return {
        "status": "complete",
        "run_id": request.run_id,
        "steps_requested": request.steps,
        "identity": accepted_identity(),
        "checkpoint_dir": str(run_root / request.run_id / "checkpoints"),
    }


__all__ = [
    "ProductionRequest",
    "accepted_identity",
    "build_training_command",
    "prepare_dataset",
    "request_from_payload",
    "request_payload",
    "run_provider_payload",
]
