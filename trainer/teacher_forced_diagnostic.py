"""Teacher-forced confidence diagnostics for held-out validation tokens."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

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


def _checkpoint_dataset_identity(
    checkpoint_manifest: Path,
) -> tuple[str, str, dict[str, object]]:
    """Return the checkpoint's dataset identity mode, expected SHA-256, and metadata."""

    try:
        payload = json.loads(checkpoint_manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError(
            f"checkpoint drive_manifest.json is not valid JSON: {checkpoint_manifest}"
        ) from error
    if not isinstance(payload, Mapping):
        raise RuntimeError("checkpoint drive_manifest.json must contain a JSON object")
    metadata = dict(payload)

    dataset_manifest_sha256 = metadata.get("dataset_manifest_sha256")
    if dataset_manifest_sha256 is not None:
        if (
            not isinstance(dataset_manifest_sha256, str)
            or len(dataset_manifest_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in dataset_manifest_sha256
            )
        ):
            raise RuntimeError(
                "checkpoint drive_manifest.json has an invalid dataset_manifest_sha256"
            )
        return "dataset_manifest", dataset_manifest_sha256, metadata

    return "drive_manifest", _sha256(checkpoint_manifest), metadata


def _auto_dataset_candidates() -> list[Path]:
    configured = os.environ.get("SMALL_LLM_DATASET_DIR")
    if configured:
        return [Path(configured).expanduser().resolve()]

    roots = [
        Path("/kaggle/input"),
        Path(
            os.environ.get(
                "SMALL_LLM_KAGGLE_WORK_ROOT",
                "/kaggle/working/small-llm",
            )
        ).expanduser()
        / "datasets",
    ]
    candidates: set[Path] = set()
    for search_root in roots:
        if not search_root.is_dir():
            continue
        for filename in ("drive_manifest.json", "manifest.json"):
            candidates.update(
                path.parent.resolve()
                for path in search_root.rglob(filename)
                if path.is_file()
            )
    return sorted(candidates, key=str)


def _stage_incremental_validation_dataset(
    *,
    checkpoint_metadata: Mapping[str, object],
    expected_manifest_sha256: str,
) -> Path | None:
    """Materialize only the frozen validation shards for a modern incremental checkpoint."""

    dataset_run_id = checkpoint_metadata.get("dataset_run_id")
    if not isinstance(dataset_run_id, str) or not dataset_run_id:
        return None

    bucket_id = os.environ.get("SMALL_LLM_HF_DATASET_BUCKET_ID", "").strip()
    if not bucket_id:
        model_repo_id = os.environ.get("SMALL_LLM_HF_REPO_ID", "").strip()
        if not model_repo_id:
            return None
        bucket_id = f"{model_repo_id}-datasets"

    from dataset.incremental_frontier import read_frontier, read_run_contract
    from dataset.incremental_stage import _stable_consumer_manifest
    from dataset.src.hf_bucket_shards import HuggingFaceBucketShardStore
    from dataset.src.storage import write_json_atomic

    token = os.environ.get("HF_TOKEN")
    store = HuggingFaceBucketShardStore(
        bucket_id,
        token=token,
        private=True,
        create_bucket=False,
    )
    contract = read_run_contract(store, run_id=dataset_run_id)
    frontier = read_frontier(store, run_id=dataset_run_id, contract=contract)
    validation_rows = frontier.get("frozen_validation_shards")
    if not isinstance(validation_rows, list) or not validation_rows:
        raise RuntimeError(
            f"incremental dataset {dataset_run_id} has no frozen validation shards"
        )

    work_root = Path(
        os.environ.get(
            "SMALL_LLM_KAGGLE_WORK_ROOT",
            "/kaggle/working/small-llm",
        )
    ).expanduser()
    destination = (
        work_root
        / "eval-datasets"
        / dataset_run_id
        / expected_manifest_sha256[:16]
    ).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    manifest = _stable_consumer_manifest(contract, frontier)
    manifest_path = destination / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    observed_sha = _sha256(manifest_path)
    if observed_sha != expected_manifest_sha256:
        raise RuntimeError(
            "incremental validation consumer manifest does not match checkpoint identity: "
            f"expected {expected_manifest_sha256}, got {observed_sha}"
        )

    for index, raw in enumerate(validation_rows):
        if not isinstance(raw, Mapping):
            raise RuntimeError(
                f"incremental validation frontier entry {index} is not an object"
            )
        filename = raw.get("filename")
        byte_size = raw.get("byte_size")
        checksum = raw.get("checksum")
        if (
            not isinstance(filename, str)
            or not filename
            or isinstance(byte_size, bool)
            or not isinstance(byte_size, int)
            or byte_size <= 0
            or not isinstance(checksum, str)
            or len(checksum) != 64
        ):
            raise RuntimeError(
                f"incremental validation frontier entry {index} is malformed"
            )
        target = destination / filename
        if (
            target.is_file()
            and not target.is_symlink()
            and target.stat().st_size == byte_size
            and _sha256(target) == checksum
        ):
            continue
        if target.exists() or target.is_symlink():
            target.unlink(missing_ok=True)
        store.download_shard(
            run_id=dataset_run_id,
            logical_name=filename,
            file_id=store.object_key(dataset_run_id, filename),
            destination=target,
            byte_size=byte_size,
            sha256=checksum,
        )

    return destination


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
    identity_mode, expected_sha, checkpoint_metadata = _checkpoint_dataset_identity(
        checkpoint_manifest
    )

    if request != "auto":
        candidates = [Path(request).expanduser().resolve()]
    else:
        candidates = _auto_dataset_candidates()

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
        if manifest.is_file():
            row["dataset_manifest_sha256"] = _sha256(manifest)
        if drive_manifest.is_file():
            row["drive_manifest_sha256"] = _sha256(drive_manifest)
        inspected.append(row)

        identity_matches = (
            row.get("dataset_manifest_sha256") == expected_sha
            if identity_mode == "dataset_manifest"
            else row.get("drive_manifest_sha256") == expected_sha
        )
        if manifest.is_file() and validation.is_dir() and identity_matches:
            matches.append(root)

    if (
        request == "auto"
        and identity_mode == "dataset_manifest"
        and not matches
    ):
        staged = _stage_incremental_validation_dataset(
            checkpoint_metadata=checkpoint_metadata,
            expected_manifest_sha256=expected_sha,
        )
        if staged is not None:
            matches = [staged]

    if identity_mode == "dataset_manifest" and request == "auto" and matches:
        # Multiple rolling-cache roots with the same manifest hash are scientifically
        # equivalent for this diagnostic because the validation inventory is frozen.
        return sorted(matches, key=str)[-1]

    if len(matches) != 1:
        raise RuntimeError(
            "teacher-forced validation requires exactly one local dataset matching the "
            f"checkpoint dataset identity ({identity_mode}); found {len(matches)}. "
            f"Inspected: {inspected}"
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


def _is_word_character(character: str) -> bool:
    return character.isalnum() or character in {"_", "'", "’"}


def _annotated_target_text(
    text: str,
    token_offsets: Sequence[int],
    target_token_index: int,
    *,
    before_chars: int = 120,
    after_chars: int = 80,
) -> str:
    """Show readable ground truth with the lexical span containing the target bracketed."""

    if not 0 <= target_token_index < len(token_offsets):
        raise ValueError("target_token_index is outside the decoded token sequence")
    if before_chars < 0 or after_chars < 0:
        raise ValueError("display context sizes must be non-negative")

    token_start = int(token_offsets[target_token_index])
    token_end = (
        int(token_offsets[target_token_index + 1])
        if target_token_index + 1 < len(token_offsets)
        else len(text)
    )
    token_start = max(0, min(token_start, len(text)))
    token_end = max(token_start, min(token_end, len(text)))

    word_positions = [
        index
        for index in range(token_start, token_end)
        if _is_word_character(text[index])
    ]
    if word_positions:
        span_start = word_positions[0]
        span_end = word_positions[-1] + 1
        while span_start > 0 and _is_word_character(text[span_start - 1]):
            span_start -= 1
        while span_end < len(text) and _is_word_character(text[span_end]):
            span_end += 1
    else:
        span_start, span_end = token_start, token_end
        while span_start < span_end and text[span_start].isspace():
            span_start += 1
        while span_end > span_start and text[span_end - 1].isspace():
            span_end -= 1
        if span_start == span_end:
            span_start, span_end = token_start, token_end

    context_start = max(0, span_start - before_chars)
    context_end = min(len(text), span_end + after_chars)
    before = text[context_start:span_start]
    target = text[span_start:span_end]
    after = text[span_end:context_end]
    snippet = before + "[" + target + "]" + after
    snippet = " ".join(snippet.split())
    if context_start > 0:
        snippet = "..." + snippet
    if context_end < len(text):
        snippet += "..."
    return snippet


def _format_probability_percent(probability: float) -> str:
    return f"{probability * 100:.6g}%"


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

                source_input_ids = batch.input_ids[sequence_index]
                source_labels = batch.labels[sequence_index]
                ground_truth_ids = torch.cat((source_input_ids, source_labels[-1:])).tolist()
                ground_truth_text, ground_truth_offsets = encoding.decode_with_offsets(
                    ground_truth_ids
                )

                input_ids = source_input_ids.unsqueeze(0).to(
                    device=device,
                    non_blocking=True,
                )
                labels = source_labels.to(device=device, non_blocking=True)
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
                                "display_text": _annotated_target_text(
                                    ground_truth_text,
                                    ground_truth_offsets,
                                    position + 1,
                                ),
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
    raw_records = [
        {key: value for key, value in row.items() if key != "display_text"}
        for row in records
    ]
    dataset_drive_manifest = dataset_root / "drive_manifest.json"
    return {
        "mode": "teacher_forced_validation",
        "dataset_root": str(dataset_root),
        "dataset_manifest_sha256": _sha256(dataset_root / "manifest.json"),
        "drive_manifest_sha256": (
            _sha256(dataset_drive_manifest) if dataset_drive_manifest.is_file() else None
        ),
        "maximum_tokens": maximum_tokens,
        "top_n": top_n,
        "summary": summary,
        "representative": _representative_rows(records),
        "tokens": raw_records,
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
            true_token = str(row.get("true_token", ""))
            top1_token = str(row.get("top1_token", ""))
            print()
            print(
                "TEXT: "
                + json.dumps(str(row.get("display_text", "")), ensure_ascii=False)
            )
            print(
                "TARGET TOKEN: "
                + json.dumps(true_token, ensure_ascii=False)
                + f"  (token {int(row.get('true_token_id', 0)):,})"
            )
            print(
                "MODEL TOP-1: "
                + json.dumps(top1_token, ensure_ascii=False)
                + f"  p={_format_probability_percent(float(row.get('top1_probability', 0.0)))}"
            )
            print(
                "TARGET: "
                + json.dumps(true_token, ensure_ascii=False)
                + f"  p={_format_probability_percent(float(row.get('true_probability', 0.0)))}, "
                + f"rank={int(row.get('true_rank', 0)):,}"
            )


def json_dumps(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True)
