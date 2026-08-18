"""Build matched atomic/textual R-SFT bundles from frozen reasoning + S0 data.

This module deliberately does not judge Gemini semantics.  Its responsibility is
mechanical dataset integrity: strict reasoning JSONL shape, the frozen uniform
skill x difficulty matrix, deterministic held-out partitioning, deterministic
10% instruction retention sampled from records the S0 model actually consumed,
and native SFT shard/bundle output for the existing trainer.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

from trainer.identity import canonical_hash
from post_training.sft.bundle import BUNDLE_SCHEMA_VERSION, verify_bundle
from post_training.sft.config import SFTDataConfig
from post_training.sft.mixture import build_atomic_blocks
from post_training.sft.schema import ConversationRecord, Split, TokenizedSFTRecord
from post_training.sft.storage import SFTDatasetWriter, StoredBlock, decode_sft_block
from post_training.sft.template import GPT2ChatTemplate, TiktokenGPT2Encoder


def _load_sibling(name: str) -> ModuleType:
    module_name = f"small_llm_rsft_{name}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load R-SFT sibling module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


generation = _load_sibling("generate")
mixture = _load_sibling("mixture")
prompts = _load_sibling("prompts")
schema = _load_sibling("schema")
serialization = _load_sibling("serialization")
tokenizer = _load_sibling("tokenizer")

DEFAULT_EXAMPLES_PER_CELL = 30
DEFAULT_HELDOUT_PER_CELL = 1
DEFAULT_OPTIMIZER_TARGET_TOKENS = 32_768
DEFAULT_CONTEXT_LENGTH = 2_048
DEFAULT_SEED = 17
REASONING_SOURCE = serialization.DEFAULT_REASONING_SOURCE
TEXTUAL_MARKERS = serialization.ReasoningMarkers(
    reasoning_start="Reasoning:\n",
    reasoning_end="\n\n",
    answer_start="Answer:\n",
)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_key(seed: int, label: str, identity: str) -> bytes:
    return hashlib.sha256(f"{seed}:{label}:{identity}".encode("utf-8")).digest()


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_reasoning_token_spec(path: Path | str) -> Any:
    """Load the compact three-string token file or full tokenizer metadata."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError(f"reasoning token spec is missing or invalid: {source}") from error
    if not isinstance(payload, Mapping):
        raise RuntimeError("reasoning token spec must be a JSON object")
    nested = payload.get(tokenizer.TOKENIZER_METADATA_KEY)
    if isinstance(nested, Mapping):
        payload = nested
    compact = {"reasoning_start", "reasoning_end", "answer_start"}
    try:
        if set(payload) == compact:
            return tokenizer.ReasoningTokenSpec(
                reasoning_start=payload["reasoning_start"],
                reasoning_end=payload["reasoning_end"],
                answer_start=payload["answer_start"],
            )
        return tokenizer.ReasoningTokenSpec.from_metadata(payload)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"invalid reasoning token spec: {error}") from error


def validate_reasoning_matrix(
    records: Sequence[Any],
    *,
    examples_per_cell: int = DEFAULT_EXAMPLES_PER_CELL,
) -> dict[str, int]:
    """Require exactly the frozen number of unique examples in every R0 cell."""

    if isinstance(examples_per_cell, bool) or not isinstance(examples_per_cell, int) or examples_per_cell <= 0:
        raise ValueError("examples_per_cell must be a positive integer")
    expected_cells = {
        (skill, difficulty)
        for skill in prompts.R0_SKILLS
        for difficulty in generation.R0_DIFFICULTIES
    }
    counts: Counter[tuple[str, str]] = Counter()
    identities: set[str] = set()
    for record in records:
        cell = (record.skill, record.difficulty)
        if cell not in expected_cells:
            raise ValueError(f"reasoning record uses unsupported cell {cell!r}")
        identity = serialization.stable_conversation_id(record)
        if identity in identities:
            raise ValueError(f"reasoning corpus contains a duplicate training identity: {identity}")
        identities.add(identity)
        counts[cell] += 1
    missing_or_wrong = {
        f"{skill}/{difficulty}": counts.get((skill, difficulty), 0)
        for skill, difficulty in sorted(expected_cells)
        if counts.get((skill, difficulty), 0) != examples_per_cell
    }
    if missing_or_wrong:
        raise ValueError(
            f"reasoning corpus is not uniform at {examples_per_cell} examples/cell: {missing_or_wrong}"
        )
    expected_total = len(expected_cells) * examples_per_cell
    if len(records) != expected_total:
        raise ValueError(f"reasoning corpus has {len(records)} records; expected {expected_total}")
    return {
        f"{skill}/{difficulty}": counts[(skill, difficulty)]
        for skill, difficulty in sorted(expected_cells)
    }


def partition_reasoning_records(
    records: Sequence[Any],
    *,
    heldout_per_cell: int = DEFAULT_HELDOUT_PER_CELL,
    seed: int = DEFAULT_SEED,
) -> dict[str, tuple[Any, ...]]:
    """Make a matched per-cell train/validation/test partition.

    For the 30-example pilot and the default heldout value this yields exactly
    28 train + 1 validation + 1 test example in every one of the 21 cells.
    """

    if isinstance(heldout_per_cell, bool) or not isinstance(heldout_per_cell, int) or heldout_per_cell <= 0:
        raise ValueError("heldout_per_cell must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    grouped: dict[tuple[str, str], list[Any]] = {}
    for record in records:
        grouped.setdefault((record.skill, record.difficulty), []).append(record)

    result: dict[str, list[Any]] = {"train": [], "validation": [], "test": []}
    for cell in sorted(grouped):
        ordered = sorted(
            grouped[cell],
            key=lambda item: _stable_key(
                seed,
                "reasoning-partition",
                serialization.stable_conversation_id(item),
            ),
        )
        if len(ordered) <= 2 * heldout_per_cell:
            raise ValueError(
                f"cell {cell!r} needs more than {2 * heldout_per_cell} examples for the requested held-out split"
            )
        result["validation"].extend(ordered[:heldout_per_cell])
        result["test"].extend(ordered[heldout_per_cell : 2 * heldout_per_cell])
        result["train"].extend(ordered[2 * heldout_per_cell :])

    for split, values in result.items():
        values.sort(
            key=lambda item: _stable_key(
                seed,
                f"reasoning-{split}-order",
                serialization.stable_conversation_id(item),
            )
        )
    return {split: tuple(values) for split, values in result.items()}


def _assert_no_atomic_marker_collision(records: Sequence[Any], spec: Any) -> None:
    tokens = tuple(spec.special_tokens)
    for record in records:
        for field in ("problem", "reasoning", "answer"):
            text = getattr(record, field)
            for token in tokens:
                if token in text:
                    raise ValueError(
                        f"reasoning record {serialization.stable_conversation_id(record)} contains reserved atomic marker text in {field}"
                    )


def _tokenize_reasoning_split(
    records: Sequence[Any],
    *,
    split: Split,
    arm: str,
    token_spec: Any,
    context_length: int,
) -> tuple[TokenizedSFTRecord, ...]:
    if arm not in {"atomic", "textual"}:
        raise ValueError("arm must be atomic or textual")
    if arm == "atomic":
        markers = serialization.ReasoningMarkers(
            reasoning_start=token_spec.reasoning_start,
            reasoning_end=token_spec.reasoning_end,
            answer_start=token_spec.answer_start,
        )
        encoder = tokenizer.ReasoningGPT2Encoder(token_spec)
    else:
        markers = TEXTUAL_MARKERS
        encoder = TiktokenGPT2Encoder()

    template = GPT2ChatTemplate(
        eos_token_id=50_256,
        maximum_context_tokens=context_length,
        # R0 has no additional assistant-only trace ceiling; the model context
        # remains the hard serialization bound.
        maximum_assistant_tokens=context_length,
    )
    result: list[TokenizedSFTRecord] = []
    special_ids = {
        tokenizer.REASONING_START_TOKEN_ID,
        tokenizer.REASONING_END_TOKEN_ID,
        tokenizer.ANSWER_START_TOKEN_ID,
    }
    for example in records:
        mapping = serialization.to_conversation_mapping(
            example,
            markers=markers,
            split=split,
        )
        conversation = ConversationRecord.from_mapping(mapping)
        tokenized = template.encode_conversation(conversation, encoder)
        counts = Counter(tokenized.token_ids)
        if arm == "atomic":
            for token_id in special_ids:
                if counts[token_id] != 1:
                    raise RuntimeError(
                        f"atomic reasoning record {tokenized.record_id} must contain control ID {token_id} exactly once"
                    )
        elif special_ids.intersection(tokenized.token_ids):
            raise RuntimeError("textual R-SFT arm unexpectedly emitted promoted reasoning token IDs")
        result.append(tokenized)
    return tuple(result)


def _read_bundle_manifest(root: Path) -> dict[str, object]:
    try:
        payload = json.loads((root / "bundle-manifest.json").read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError(f"S0 bundle manifest is missing or invalid: {root}") from error
    if not isinstance(payload, Mapping):
        raise RuntimeError("S0 bundle manifest must be a JSON object")
    return dict(payload)


def _instruction_source_shares(s0_bundle_manifest: Mapping[str, object]) -> dict[str, float]:
    raw = s0_bundle_manifest.get("instruction_source_shares")
    if not isinstance(raw, Mapping) or not raw:
        raise RuntimeError("S0 bundle does not record instruction_source_shares")
    result: dict[str, float] = {}
    for source, value in raw.items():
        if not isinstance(source, str) or not source:
            raise RuntimeError("S0 instruction source name is malformed")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(f"S0 instruction share for {source!r} is malformed")
        share = float(value)
        if not math.isfinite(share) or share <= 0:
            raise RuntimeError(f"S0 instruction share for {source!r} must be positive")
        result[source] = share
    total = sum(result.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError("S0 instruction_source_shares do not sum to one")
    return result


def iter_tokenized_bundle_records(root: Path | str, *, split: str = "train") -> Iterable[TokenizedSFTRecord]:
    """Yield exact stored records from a verified SFT split without re-tokenizing."""

    bundle = Path(root)
    split_root = bundle / split
    try:
        manifest = json.loads((split_root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError(f"stored SFT split manifest is missing or invalid: {split_root}") from error
    if not isinstance(manifest, Mapping):
        raise RuntimeError("stored SFT split manifest must be an object")
    raw_blocks = manifest.get("blocks")
    if not isinstance(raw_blocks, list):
        raise RuntimeError("stored SFT split manifest has no blocks")
    blocks = sorted(
        (StoredBlock.from_mapping(item) for item in raw_blocks if isinstance(item, Mapping) and item.get("split") == split),
        key=lambda item: item.block_id,
    )
    for stored in blocks:
        path = split_root / stored.shard
        with path.open("rb") as handle:
            handle.seek(stored.offset)
            payload = handle.read(stored.byte_size)
        if len(payload) != stored.byte_size:
            raise RuntimeError(f"short read from S0 shard {path}")
        block = decode_sft_block(payload)
        if block.block_id != stored.block_id or block.target_token_count != stored.target_token_count:
            raise RuntimeError("stored S0 block identity drifted after bundle verification")
        yield from block.records


def retention_target_tokens_for_matched_arms(
    atomic_reasoning_targets: int,
    textual_reasoning_targets: int,
) -> int:
    """Choose one arm-neutral retention target so both ablation arms reuse identical S0 records."""

    for name, value in (
        ("atomic_reasoning_targets", atomic_reasoning_targets),
        ("textual_reasoning_targets", textual_reasoning_targets),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    reference = round((atomic_reasoning_targets + textual_reasoning_targets) / 2)
    return max(1, round(reference * mixture.RETENTION_SHARE / mixture.REASONING_SHARE))


def select_s0_retention_records(
    records: Iterable[TokenizedSFTRecord],
    *,
    source_shares: Mapping[str, float],
    target_tokens: int,
) -> tuple[TokenizedSFTRecord, ...]:
    """Take a deterministic no-replacement S0 instruction slice at source-token quotas.

    The S0 training stream is already independently shuffled per instruction
    source before source mixing.  Taking each source's next records therefore
    samples from the exact tokenized examples the model saw without reconstructing
    or re-tokenizing upstream SmolTalk.
    """

    if isinstance(target_tokens, bool) or not isinstance(target_tokens, int) or target_tokens <= 0:
        raise ValueError("target_tokens must be a positive integer")
    normalized = {str(source): float(share) for source, share in source_shares.items()}
    if not normalized or any(share <= 0 for share in normalized.values()):
        raise ValueError("source_shares must contain positive entries")
    if not math.isclose(sum(normalized.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("source_shares must sum to one")

    quotas = {source: target_tokens * share for source, share in normalized.items()}
    selected_tokens = {source: 0 for source in normalized}
    selected: list[TokenizedSFTRecord] = []
    for record in records:
        source = record.source
        if source not in normalized or selected_tokens[source] >= quotas[source]:
            continue
        if any(token_id >= tokenizer.BASE_SEMANTIC_VOCAB_SIZE for token_id in record.token_ids):
            raise RuntimeError(
                f"S0 retention record {record.record_id} contains a non-S0 semantic token ID"
            )
        selected.append(record)
        selected_tokens[source] += record.target_token_count
        if all(selected_tokens[name] >= quotas[name] for name in normalized):
            break

    missing = {
        source: {"selected": selected_tokens[source], "target": quotas[source]}
        for source in normalized
        if selected_tokens[source] < quotas[source]
    }
    if missing:
        raise RuntimeError(f"S0 instruction bundle exhausted before retention quotas were met: {missing}")
    return tuple(selected)


def _common_record_order(
    records: Sequence[TokenizedSFTRecord],
    *,
    seed: int,
    split: str,
) -> tuple[TokenizedSFTRecord, ...]:
    identities = [record.record_id for record in records]
    if len(set(identities)) != len(identities):
        raise ValueError(f"R-SFT {split} stream contains duplicate record IDs")
    return tuple(
        sorted(records, key=lambda item: _stable_key(seed, f"bundle-{split}-order", item.record_id))
    )


def _write_split(
    output_dir: Path,
    *,
    records: Sequence[TokenizedSFTRecord],
    split: Split,
    arm: str,
    requested_source_shares: Mapping[str, float],
    optimizer_target_tokens: int,
    context_length: int,
    seed: int,
) -> dict[str, object]:
    if not records:
        raise RuntimeError(f"cannot build empty R-SFT {split} split")
    actual_targets = sum(record.target_token_count for record in records)
    config = SFTDataConfig(
        target_loss_tokens=actual_targets,
        optimizer_target_tokens=optimizer_target_tokens,
        context_length=context_length,
        maximum_assistant_tokens=context_length,
        instruction_share=1.0,
        replay_share=0.0,
        instruction_source_shares=dict(requested_source_shares),
        seed=seed,
    )
    blocks = build_atomic_blocks(records, target_tokens_per_block=optimizer_target_tokens)
    manifest = SFTDatasetWriter(output_dir, config).write(blocks)
    report_without_hash: dict[str, object] = {
        "schema": "small-llm-rsft-build-report-v1",
        "manifest_identity": manifest["manifest_sha256"],
        "arm": arm,
        "split": split,
        "records": len(records),
        "loss_bearing_target_tokens": actual_targets,
        "requested_source_shares": dict(requested_source_shares),
        "actual_source_target_tokens": manifest["totals"]["source_target_tokens"],  # type: ignore[index]
    }
    report = {**report_without_hash, "report_sha256": canonical_hash(report_without_hash)}
    (output_dir / "build-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"manifest": manifest, "report": report}


def _source_manifest(
    *,
    reasoning_path: Path,
    reasoning_records: Sequence[Any],
    cells: Mapping[str, int],
    partition: Mapping[str, Sequence[Any]],
    s0_verification: Mapping[str, object],
    s0_source_shares: Mapping[str, float],
    retention_records: Sequence[TokenizedSFTRecord],
    retention_target_requested: int,
    heldout_per_cell: int,
    seed: int,
) -> dict[str, object]:
    retention_tokens: Counter[str] = Counter()
    for record in retention_records:
        retention_tokens[record.source] += record.target_token_count
    without_hash: dict[str, object] = {
        "schema": "small-llm-rsft-source-v1",
        "reasoning": {
            "path_name": reasoning_path.name,
            "sha256": _sha256_path(reasoning_path),
            "byte_size": reasoning_path.stat().st_size,
            "records": len(reasoning_records),
            "cells": dict(cells),
        },
        "partition": {
            "policy": "per-cell-stable-hash",
            "heldout_per_cell_per_split": heldout_per_cell,
            "seed": seed,
            "records": {
                split: [serialization.stable_conversation_id(record) for record in values]
                for split, values in partition.items()
            },
        },
        "retention": {
            "source": "completed-s0-tokenized-train-split",
            "s0_bundle_manifest_sha256": s0_verification["bundle_manifest_sha256"],
            "requested_top_level_share": mixture.RETENTION_SHARE,
            "source_shares": dict(s0_source_shares),
            "requested_target_tokens": retention_target_requested,
            "selected_target_tokens": sum(record.target_token_count for record in retention_records),
            "selected_source_target_tokens": dict(sorted(retention_tokens.items())),
            "record_ids": [record.record_id for record in retention_records],
        },
    }
    return {**without_hash, "manifest_sha256": canonical_hash(without_hash)}


def _build_arm(
    output_dir: Path,
    *,
    arm: str,
    partition: Mapping[str, Sequence[Any]],
    token_spec: Any,
    retention_records: Sequence[TokenizedSFTRecord],
    requested_train_source_shares: Mapping[str, float],
    source_manifest: Mapping[str, object],
    optimizer_target_tokens: int,
    context_length: int,
    seed: int,
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to replace existing R-SFT arm bundle: {output_dir}")
    temporary = output_dir.with_name(f".{output_dir.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    split_rows: dict[str, object] = {}
    try:
        for split in ("train", "validation", "test"):
            typed_split: Split = split  # type: ignore[assignment]
            reasoning_tokenized = _tokenize_reasoning_split(
                partition[split],
                split=typed_split,
                arm=arm,
                token_spec=token_spec,
                context_length=context_length,
            )
            if split == "train":
                combined = _common_record_order(
                    (*reasoning_tokenized, *retention_records),
                    seed=seed,
                    split=split,
                )
                shares = requested_train_source_shares
            else:
                combined = _common_record_order(reasoning_tokenized, seed=seed, split=split)
                shares = {REASONING_SOURCE: 1.0}
            result = _write_split(
                temporary / split,
                records=combined,
                split=typed_split,
                arm=arm,
                requested_source_shares=shares,
                optimizer_target_tokens=optimizer_target_tokens,
                context_length=context_length,
                seed=seed,
            )
            split_rows[split] = {
                "path": split,
                "manifest_sha256": result["manifest"]["manifest_sha256"],
                "loss_bearing_target_tokens": result["manifest"]["totals"]["loss_bearing_target_tokens"],
                "build_report_sha256": result["report"]["report_sha256"],
            }

        (temporary / "source-manifest.json").write_text(
            json.dumps(dict(source_manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        token_metadata = token_spec.to_metadata()
        (temporary / "reasoning-tokens.json").write_text(
            json.dumps(token_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        train_targets = int(split_rows["train"]["loss_bearing_target_tokens"])  # type: ignore[index]
        bundle_without_hash: dict[str, object] = {
            "schema": "small-llm-sft-bundle",
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "prepared_source_manifest_sha256": source_manifest["manifest_sha256"],
            "prepared_source": {
                "dataset_name": "small-llm-rsft-r0-pilot",
                "revision": source_manifest["reasoning"]["sha256"],  # type: ignore[index]
                "license_id": "synthetic-plus-s0-retention",
                "split_policy": source_manifest["partition"],
            },
            "train_target_tokens_requested": train_targets,
            "optimizer_target_tokens": optimizer_target_tokens,
            "context_length": context_length,
            "maximum_assistant_tokens": context_length,
            "instruction_share": 1.0,
            "replay_share": 0.0,
            "instruction_source_shares": dict(requested_train_source_shares),
            "seed": seed,
            "splits": split_rows,
            "rsft": {
                "stage": "r_sft_r0",
                "delimiter_format": arm,
                "reasoning_share_requested": mixture.REASONING_SHARE,
                "retention_share_requested": mixture.RETENTION_SHARE,
                "reasoning_tokenizer": token_metadata,
                "textual_markers": {
                    "reasoning_start": TEXTUAL_MARKERS.reasoning_start,
                    "reasoning_end": TEXTUAL_MARKERS.reasoning_end,
                    "answer_start": TEXTUAL_MARKERS.answer_start,
                }
                if arm == "textual"
                else None,
            },
        }
        bundle_manifest = {
            **bundle_without_hash,
            "manifest_sha256": canonical_hash(bundle_without_hash),
        }
        _atomic_write_json(temporary / "bundle-manifest.json", bundle_manifest)
        temporary.rename(output_dir)
        verification = verify_bundle(output_dir)
        return {
            "bundle": bundle_manifest,
            "verification": verification,
            "train_source_target_tokens": json.loads(
                (output_dir / "train" / "manifest.json").read_text(encoding="utf-8")
            )["totals"]["source_target_tokens"],
        }
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_matched_pilot_bundles(
    reasoning_jsonl: Path | str,
    *,
    s0_bundle: Path | str,
    token_spec_path: Path | str,
    output_dir: Path | str,
    examples_per_cell: int = DEFAULT_EXAMPLES_PER_CELL,
    heldout_per_cell: int = DEFAULT_HELDOUT_PER_CELL,
    optimizer_target_tokens: int = DEFAULT_OPTIMIZER_TARGET_TOKENS,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    """Freeze one reasoning pilot into matched atomic/textual native SFT bundles."""

    reasoning_path = Path(reasoning_jsonl).expanduser().resolve()
    source_bundle = Path(s0_bundle).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing matched R-SFT bundle root: {output}")
    if not reasoning_path.is_file() or reasoning_path.is_symlink():
        raise RuntimeError(f"reasoning JSONL is missing or unsafe: {reasoning_path}")
    if not source_bundle.is_dir():
        raise RuntimeError(f"S0 bundle directory is missing: {source_bundle}")
    if optimizer_target_tokens <= 0 or context_length <= 0:
        raise ValueError("optimizer_target_tokens and context_length must be positive")

    records = schema.read_jsonl(reasoning_path)
    cells = validate_reasoning_matrix(records, examples_per_cell=examples_per_cell)
    partition = partition_reasoning_records(
        records,
        heldout_per_cell=heldout_per_cell,
        seed=seed,
    )
    token_spec = load_reasoning_token_spec(token_spec_path)
    _assert_no_atomic_marker_collision(records, token_spec)

    s0_verification = verify_bundle(source_bundle)
    s0_manifest = _read_bundle_manifest(source_bundle)
    retention_source_shares = _instruction_source_shares(s0_manifest)

    atomic_train = _tokenize_reasoning_split(
        partition["train"],
        split="train",
        arm="atomic",
        token_spec=token_spec,
        context_length=context_length,
    )
    textual_train = _tokenize_reasoning_split(
        partition["train"],
        split="train",
        arm="textual",
        token_spec=token_spec,
        context_length=context_length,
    )
    atomic_reasoning_targets = sum(record.target_token_count for record in atomic_train)
    textual_reasoning_targets = sum(record.target_token_count for record in textual_train)
    retention_target = retention_target_tokens_for_matched_arms(
        atomic_reasoning_targets,
        textual_reasoning_targets,
    )
    retention_records = select_s0_retention_records(
        iter_tokenized_bundle_records(source_bundle, split="train"),
        source_shares=retention_source_shares,
        target_tokens=retention_target,
    )
    requested_train_source_shares = mixture.build_rsft_source_shares(retention_source_shares)
    source_manifest = _source_manifest(
        reasoning_path=reasoning_path,
        reasoning_records=records,
        cells=cells,
        partition=partition,
        s0_verification=s0_verification,
        s0_source_shares=retention_source_shares,
        retention_records=retention_records,
        retention_target_requested=retention_target,
        heldout_per_cell=heldout_per_cell,
        seed=seed,
    )

    output.mkdir(parents=True)
    try:
        (output / "source-manifest.json").write_text(
            json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        shutil.copy2(reasoning_path, output / "reasoning.jsonl")
        atomic = _build_arm(
            output / "atomic",
            arm="atomic",
            partition=partition,
            token_spec=token_spec,
            retention_records=retention_records,
            requested_train_source_shares=requested_train_source_shares,
            source_manifest=source_manifest,
            optimizer_target_tokens=optimizer_target_tokens,
            context_length=context_length,
            seed=seed,
        )
        textual = _build_arm(
            output / "textual",
            arm="textual",
            partition=partition,
            token_spec=token_spec,
            retention_records=retention_records,
            requested_train_source_shares=requested_train_source_shares,
            source_manifest=source_manifest,
            optimizer_target_tokens=optimizer_target_tokens,
            context_length=context_length,
            seed=seed,
        )

        selected_retention_targets = sum(record.target_token_count for record in retention_records)
        arm_rows: dict[str, object] = {}
        for name, row, reasoning_targets in (
            ("atomic", atomic, atomic_reasoning_targets),
            ("textual", textual, textual_reasoning_targets),
        ):
            train_total = int(row["bundle"]["train_target_tokens_requested"])  # type: ignore[index]
            arm_rows[name] = {
                "bundle_path": name,
                "bundle_manifest_sha256": row["bundle"]["manifest_sha256"],  # type: ignore[index]
                "train_target_tokens": train_total,
                "reasoning_train_target_tokens": reasoning_targets,
                "retention_train_target_tokens": selected_retention_targets,
                "realized_retention_share": selected_retention_targets / train_total,
                "train_source_target_tokens": row["train_source_target_tokens"],
            }
        pilot_without_hash: dict[str, object] = {
            "schema": "small-llm-rsft-pilot-bundles-v1",
            "source_manifest_sha256": source_manifest["manifest_sha256"],
            "examples": {
                "total": len(records),
                "examples_per_cell": examples_per_cell,
                "train": len(partition["train"]),
                "validation": len(partition["validation"]),
                "test": len(partition["test"]),
            },
            "retention": {
                "requested_share": mixture.RETENTION_SHARE,
                "reference_target_tokens": retention_target,
                "selected_target_tokens": selected_retention_targets,
                "records": len(retention_records),
                "source_shares": retention_source_shares,
            },
            "arms": arm_rows,
            "seed": seed,
        }
        pilot_manifest = {
            **pilot_without_hash,
            "manifest_sha256": canonical_hash(pilot_without_hash),
        }
        _atomic_write_json(output / "pilot-manifest.json", pilot_manifest)
        return pilot_manifest
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise


__all__ = [
    "DEFAULT_CONTEXT_LENGTH",
    "DEFAULT_EXAMPLES_PER_CELL",
    "DEFAULT_HELDOUT_PER_CELL",
    "DEFAULT_OPTIMIZER_TARGET_TOKENS",
    "DEFAULT_SEED",
    "TEXTUAL_MARKERS",
    "build_matched_pilot_bundles",
    "iter_tokenized_bundle_records",
    "load_reasoning_token_spec",
    "partition_reasoning_records",
    "retention_target_tokens_for_matched_arms",
    "select_s0_retention_records",
    "validate_reasoning_matrix",
]
