"""Teacher-forced confidence diagnostics for held-out validation tokens."""

from __future__ import annotations

import hashlib
import math
import os
import statistics
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .precision import autocast_context
from .shards import SchemaV2ShardReader

DEFAULT_TEACHER_FORCED_TOKENS = 4_096
_METRIC_POSITION_CHUNK = 256


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_validation_dataset(
    request: str,
    *,
    checkpoint_root: Path,
) -> Path:
    """Resolve and identity-check the local validation dataset for a checkpoint."""

    checkpoint_manifest = checkpoint_root / "drive_manifest.json"
    if not checkpoint_manifest.is_file():
        raise RuntimeError(
            "the checkpoint has no drive_manifest.json; cannot prove validation-dataset identity"
        )
    expected_sha = _sha256(checkpoint_manifest)

    if request != "auto":
        candidates = [Path(request).expanduser().resolve()]
    else:
        configured = os.environ.get("SMALL_LLM_DATASET_DIR")
        if configured:
            candidates = [Path(configured).expanduser().resolve()]
        else:
            kaggle_input = Path("/kaggle/input")
            candidates = (
                sorted({path.parent for path in kaggle_input.rglob("drive_manifest.json")})
                if kaggle_input.is_dir()
                else []
            )

    matches: list[Path] = []
    inspected: list[dict[str, object]] = []
    for root in candidates:
        drive_manifest = root / "drive_manifest.json"
        manifest = root / "manifest.json"
        validation = root / "validation"
        row: dict[str, object] = {
            "root": str(root),
            "manifest": manifest.is_file(),
            "drive_manifest": drive_manifest.is_file(),
            "validation": validation.is_dir(),
        }
        if drive_manifest.is_file():
            row["drive_manifest_sha256"] = _sha256(drive_manifest)
        inspected.append(row)
        if (
            manifest.is_file()
            and validation.is_dir()
            and row.get("drive_manifest_sha256") == expected_sha
        ):
            matches.append(root)

    if len(matches) != 1:
        raise RuntimeError(
            "teacher-forced validation requires exactly one local dataset matching the "
            f"checkpoint drive manifest; found {len(matches)}. Inspected: {inspected}"
        )
    return matches[0]


def teacher_forced_token_metrics(
    logits: Tensor,
    labels: Tensor,
    *,
    top_n: int = 5,
) -> dict[str, Tensor]:
    """Compute raw-distribution diagnostics for a rank-2 logits/labels slice."""

    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
        raise ValueError("logits must be [positions, vocab] and labels must be [positions]")
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    active = labels.ne(-100)
    scores = logits[active].float()
    targets = labels[active].long()
    if targets.numel() == 0:
        return {
            "active_indices": active.nonzero(as_tuple=False).flatten(),
            "true_log_probability": torch.empty(0),
            "true_probability": torch.empty(0),
            "true_rank": torch.empty(0, dtype=torch.long),
            "top_token_ids": torch.empty((0, 0), dtype=torch.long),
            "top_probabilities": torch.empty((0, 0)),
            "top5_mass": torch.empty(0),
            "entropy": torch.empty(0),
        }
    if bool((targets < 0).any()) or bool((targets >= scores.shape[-1]).any()):
        raise ValueError("labels contain token IDs outside the model vocabulary")
    if not bool(torch.isfinite(scores).all()):
        raise FloatingPointError("teacher-forced logits contain non-finite values")

    log_normalizer = torch.logsumexp(scores, dim=-1)
    true_logits = scores.gather(1, targets[:, None]).squeeze(1)
    true_log_probability = true_logits - log_normalizer
    true_probability = true_log_probability.exp()

    count = min(top_n, scores.shape[-1])
    top_logits, top_token_ids = torch.topk(scores, count, dim=-1)
    top_probabilities = (top_logits - log_normalizer[:, None]).exp()
    top5_mass = top_probabilities[:, : min(5, count)].sum(dim=-1)

    probabilities = torch.softmax(scores, dim=-1)
    entropy = log_normalizer - (probabilities * scores).sum(dim=-1)
    del probabilities

    true_rank = scores.gt(true_logits[:, None]).sum(dim=-1) + 1
    return {
        "active_indices": active.nonzero(as_tuple=False).flatten(),
        "true_log_probability": true_log_probability,
        "true_probability": true_probability,
        "true_rank": true_rank,
        "top_token_ids": top_token_ids,
        "top_probabilities": top_probabilities,
        "top5_mass": top5_mass,
        "entropy": entropy,
    }


def _decode_token(encoding: Any, token_id: int) -> str:
    return encoding.decode_single_token_bytes(token_id).decode("utf-8", errors="replace")


def _summary(records: list[dict[str, object]]) -> dict[str, float | int]:
    if not records:
        raise RuntimeError("teacher-forced diagnostic produced no active target tokens")

    true_log_probabilities = [float(row["true_log_probability"]) for row in records]
    true_probabilities = [float(row["true_probability"]) for row in records]
    top1_probabilities = [float(row["top1_probability"]) for row in records]
    ranks = [int(row["true_rank"]) for row in records]
    entropies = [float(row["entropy"]) for row in records]
    top5_masses = [float(row["top5_mass"]) for row in records]
    count = len(records)
    mean_loss = -sum(true_log_probabilities) / count
    wrong = [row for row in records if int(row["true_rank"]) != 1]
    confidently_wrong = [
        row for row in wrong if float(row["top1_probability"]) >= 0.5
    ]

    return {
        "target_tokens": count,
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 80.0)),
        "mean_true_probability": statistics.fmean(true_probabilities),
        "median_true_probability": statistics.median(true_probabilities),
        "mean_top1_probability": statistics.fmean(top1_probabilities),
        "median_top1_probability": statistics.median(top1_probabilities),
        "top1_accuracy": sum(rank == 1 for rank in ranks) / count,
        "true_rank_le_5": sum(rank <= 5 for rank in ranks) / count,
        "true_rank_le_10": sum(rank <= 10 for rank in ranks) / count,
        "true_rank_le_100": sum(rank <= 100 for rank in ranks) / count,
        "median_true_rank": statistics.median(ranks),
        "mean_entropy_nats": statistics.fmean(entropies),
        "mean_top5_mass": statistics.fmean(top5_masses),
        "top1_probability_lt_0_1": sum(value < 0.1 for value in top1_probabilities) / count,
        "confidently_wrong_ge_0_5": len(confidently_wrong) / count,
    }


def _representative_rows(
    records: list[dict[str, object]],
    *,
    limit: int = 5,
) -> dict[str, list[dict[str, object]]]:
    lowest_true = sorted(records, key=lambda row: float(row["true_probability"]))[:limit]
    confidently_wrong = sorted(
        (row for row in records if int(row["true_rank"]) != 1),
        key=lambda row: float(row["top1_probability"]),
        reverse=True,
    )[:limit]
    return {
        "lowest_true_probability": lowest_true,
        "highest_confidence_wrong": confidently_wrong,
    }


@torch.inference_mode()
def run_teacher_forced_validation(
    model: nn.Module,
    *,
    model_config: object,
    checkpoint_root: Path,
    dataset_request: str,
    device: torch.device,
    precision: str,
    encoding: Any,
    maximum_tokens: int = DEFAULT_TEACHER_FORCED_TOKENS,
    top_n: int = 5,
) -> dict[str, object]:
    """Run deterministic teacher-forced diagnostics over held-out validation text."""

    if maximum_tokens <= 0:
        raise ValueError("maximum_tokens must be positive")
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    dataset_root = resolve_validation_dataset(dataset_request, checkpoint_root=checkpoint_root)
    max_seq_len = int(getattr(model_config, "max_seq_len"))
    semantic_vocab_size = int(getattr(model_config, "semantic_vocab_size"))
    reader = SchemaV2ShardReader(
        dataset_root,
        split="validation",
        semantic_vocab_size=semantic_vocab_size,
        context_length=max_seq_len,
    )

    was_training = model.training
    model.eval()
    records: list[dict[str, object]] = []
    try:
        for batch in reader.iter_from_start():
            if batch.split != "validation":
                raise RuntimeError("validation reader yielded a non-validation batch")
            for sequence_index in range(batch.sequence_count):
                if len(records) >= maximum_tokens:
                    break
                input_ids = batch.input_ids[sequence_index : sequence_index + 1].to(
                    device=device,
                    non_blocking=True,
                )
                labels = batch.labels[sequence_index].to(device=device, non_blocking=True)
                with autocast_context(precision, device):
                    logits = model(input_ids)[0]

                remaining = maximum_tokens - len(records)
                active_positions = labels.ne(-100).nonzero(as_tuple=False).flatten()[:remaining]
                for start in range(0, int(active_positions.numel()), _METRIC_POSITION_CHUNK):
                    positions = active_positions[start : start + _METRIC_POSITION_CHUNK]
                    if positions.numel() == 0:
                        continue
                    metrics = teacher_forced_token_metrics(
                        logits.index_select(0, positions),
                        labels.index_select(0, positions),
                        top_n=top_n,
                    )
                    active_indices = metrics["active_indices"].tolist()
                    true_log_probability = metrics["true_log_probability"].tolist()
                    true_probability = metrics["true_probability"].tolist()
                    true_rank = metrics["true_rank"].tolist()
                    top_token_ids = metrics["top_token_ids"].tolist()
                    top_probabilities = metrics["top_probabilities"].tolist()
                    top5_mass = metrics["top5_mass"].tolist()
                    entropy = metrics["entropy"].tolist()

                    for local_index, metric_index in enumerate(active_indices):
                        position = int(positions[int(metric_index)].item())
                        true_token_id = int(labels[position].item())
                        candidate_ids = [int(value) for value in top_token_ids[local_index]]
                        candidate_probabilities = [
                            float(value) for value in top_probabilities[local_index]
                        ]
                        context_start = max(0, position - 23)
                        context_ids = input_ids[0, context_start : position + 1].tolist()
                        candidates = [
                            {
                                "rank": rank,
                                "token_id": token_id,
                                "token": _decode_token(encoding, token_id),
                                "probability": probability,
                            }
                            for rank, (token_id, probability) in enumerate(
                                zip(candidate_ids, candidate_probabilities, strict=True),
                                start=1,
                            )
                        ]
                        records.append(
                            {
                                "block_id": int(batch.block_id),
                                "sequence_index": sequence_index,
                                "position": position,
                                "context_tail": encoding.decode(context_ids),
                                "true_token_id": true_token_id,
                                "true_token": _decode_token(encoding, true_token_id),
                                "true_log_probability": float(true_log_probability[local_index]),
                                "true_probability": float(true_probability[local_index]),
                                "true_rank": int(true_rank[local_index]),
                                "top1_token_id": candidate_ids[0],
                                "top1_token": _decode_token(encoding, candidate_ids[0]),
                                "top1_probability": candidate_probabilities[0],
                                "top5_mass": float(top5_mass[local_index]),
                                "entropy": float(entropy[local_index]),
                                "top_tokens": candidates,
                            }
                        )
                        if len(records) >= maximum_tokens:
                            break
                    del metrics
                    if len(records) >= maximum_tokens:
                        break
                del logits, input_ids, labels
                if len(records) >= maximum_tokens:
                    break
            if len(records) >= maximum_tokens:
                break
    finally:
        model.train(was_training)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = _summary(records)
    return {
        "mode": "teacher_forced_validation",
        "dataset_root": str(dataset_root),
        "drive_manifest_sha256": _sha256(dataset_root / "drive_manifest.json"),
        "maximum_tokens": maximum_tokens,
        "top_n": top_n,
        "summary": summary,
        "representative": _representative_rows(records),
        "tokens": records,
    }


def print_teacher_forced_report(report: Mapping[str, object]) -> None:
    summary = report.get("summary")
    representative = report.get("representative")
    if not isinstance(summary, Mapping) or not isinstance(representative, Mapping):
        raise RuntimeError("teacher-forced report has an invalid structure")

    print("\n" + "=" * 80)
    print("Teacher-forced held-out confidence diagnostic")
    print(json_dumps(summary))
    for label in ("lowest_true_probability", "highest_confidence_wrong"):
        rows = representative.get(label)
        if not isinstance(rows, list):
            continue
        print(f"\n{label}:")
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            print(
                f"context={str(row.get('context_tail'))!r} true={str(row.get('true_token'))!r} "
                f"p_true={float(row.get('true_probability', 0.0)):.6f} "
                f"rank={int(row.get('true_rank', 0))} "
                f"top1={str(row.get('top1_token'))!r} "
                f"p_top1={float(row.get('top1_probability', 0.0)):.6f}"
            )


def json_dumps(value: object) -> str:
    import json

    return json.dumps(value, indent=2, sort_keys=True)
