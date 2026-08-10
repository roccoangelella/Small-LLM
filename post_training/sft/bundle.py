"""Deterministic SFT source preparation, identity splitting, and bundle construction."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Iterable, Iterator, Mapping

from dataset.src.remote import sha256_path
from trainer.identity import canonical_hash

from .behavior_eval import behavior_prompt_texts
from .builder import SFTDatasetBuilder
from .config import DEFAULT_INSTRUCTION_SOURCE_SHARES, SFTDataConfig
from .schema import ChatMessage, ConversationRecord, Split, TokenizedSFTRecord
from .sources import JsonlConversationSource, iter_schema_v2_replay
from .storage import SFTShardReader
from .template import TiktokenGPT2Encoder

SMOL_SMOLTALK_DATASET = "HuggingFaceTB/smol-smoltalk"
SMOL_SMOLTALK_DATA_REVISION = "f80219b491a28e79600fa320e075752f1ea0303e"
SMOL_SMOLTALK_LICENSE = "apache-2.0"
SPLIT_POLICY_VERSION = 1
BUNDLE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class IdentitySplitPolicy:
    train_share: float = 0.95
    validation_share: float = 0.025
    test_share: float = 0.025
    buckets: int = 10_000
    seed: int = 17

    def __post_init__(self) -> None:
        shares = (self.train_share, self.validation_share, self.test_share)
        if any(not math.isfinite(value) or value <= 0 for value in shares):
            raise ValueError("split shares must be finite and positive")
        if not math.isclose(sum(shares), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("split shares must sum to one")
        if self.buckets <= 0:
            raise ValueError("buckets must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

    def assign(self, group_id: str) -> Split:
        digest = hashlib.sha256(f"{self.seed}:{group_id}".encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big") % self.buckets
        train_end = round(self.train_share * self.buckets)
        validation_end = round((self.train_share + self.validation_share) * self.buckets)
        if bucket < train_end:
            return "train"
        if bucket < validation_end:
            return "validation"
        return "test"

    def as_dict(self) -> dict[str, object]:
        return {
            "version": SPLIT_POLICY_VERSION,
            "train": self.train_share,
            "validation": self.validation_share,
            "test": self.test_share,
            "buckets": self.buckets,
            "seed": self.seed,
        }


def _canonical_messages(messages: Iterable[ChatMessage], *, assistant: bool) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for message in messages:
        if not assistant and message.role == "assistant":
            continue
        result.append(
            {
                "role": message.role,
                "content": "\n".join(line.rstrip() for line in message.content.strip().splitlines()),
            }
        )
    return result


def conversation_content_hash(record: ConversationRecord) -> str:
    return canonical_hash(
        {"source": record.source, "messages": _canonical_messages(record.messages, assistant=True)}
    )


def conversation_group_id(record: ConversationRecord) -> str:
    """Group prompt derivatives before splitting by excluding assistant labels."""

    return canonical_hash(
        {"source": record.source, "context": _canonical_messages(record.messages, assistant=False)}
    )


def _normalized_probe(text: str) -> str:
    return " ".join(text.casefold().split())


_DECONTAMINATION_TEXTS = tuple(
    value for value in (_normalized_probe(text) for text in behavior_prompt_texts()) if value
)


def _contaminates_behavior_suite(record: ConversationRecord) -> bool:
    non_assistant = "\n".join(
        message.content for message in record.messages if message.role != "assistant"
    )
    normalized = _normalized_probe(non_assistant)
    return any(
        normalized == probe or (len(probe) >= 32 and probe in normalized)
        for probe in _DECONTAMINATION_TEXTS
    )


def _record_payload(record: ConversationRecord) -> dict[str, object]:
    return {
        "conversation_id": record.conversation_id,
        "source": record.source,
        "split": record.split,
        "messages": [{"role": message.role, "content": message.content} for message in record.messages],
        "metadata": dict(record.metadata),
    }


def prepare_smoltalk(
    output_dir: Path | str,
    *,
    dataset_name: str = SMOL_SMOLTALK_DATASET,
    revision: str = SMOL_SMOLTALK_DATA_REVISION,
    upstream_split: str = "train",
    split_policy: IdentitySplitPolicy | None = None,
    allowed_sources: Iterable[str] = DEFAULT_INSTRUCTION_SOURCE_SHARES,
) -> dict[str, object]:
    """Export a pinned upstream split into deterministic identity-safe JSONL files."""

    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover
        raise RuntimeError(
            "SFT source preparation requires `datasets`; launch with the Kaggle SFT prepare command"
        ) from error

    root = Path(output_dir)
    if root.exists():
        raise FileExistsError(f"refusing to replace existing prepared source bundle: {root}")
    temporary = root.with_name(f".{root.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    policy = split_policy or IdentitySplitPolicy()
    allowed = frozenset(str(value) for value in allowed_sources)
    if not allowed:
        raise ValueError("allowed_sources cannot be empty")

    handles: dict[tuple[str, str], object] = {}
    counts: dict[str, dict[str, int]] = {
        source: {"train": 0, "validation": 0, "test": 0}
        for source in sorted(allowed)
    }
    seen_content: set[str] = set()
    seen = accepted = duplicates = contaminated = malformed = unsupported_source = 0

    try:
        for source in sorted(allowed):
            for split in ("train", "validation", "test"):
                path = temporary / "raw" / split / f"{source}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                handles[(source, split)] = path.open("w", encoding="utf-8")

        dataset = load_dataset(
            dataset_name,
            split=upstream_split,
            revision=revision,
            streaming=True,
        )
        for index, raw in enumerate(dataset):
            seen += 1
            if not isinstance(raw, Mapping):
                malformed += 1
                continue
            source = raw.get("source")
            if not isinstance(source, str) or source not in allowed:
                unsupported_source += 1
                continue
            try:
                base = ConversationRecord.from_mapping(
                    {
                        "conversation_id": f"{dataset_name}@{revision}:{upstream_split}:{index}",
                        "source": source,
                        "messages": raw.get("messages"),
                    }
                )
            except (TypeError, ValueError):
                malformed += 1
                continue

            content_hash = conversation_content_hash(base)
            if content_hash in seen_content:
                duplicates += 1
                continue
            seen_content.add(content_hash)
            if _contaminates_behavior_suite(base):
                contaminated += 1
                continue

            group_id = conversation_group_id(base)
            split = policy.assign(group_id)
            record = replace(
                base,
                conversation_id=f"smol-smoltalk:{content_hash}",
                split=split,
                metadata={
                    "dataset_name": dataset_name,
                    "source_revision": revision,
                    "license_id": SMOL_SMOLTALK_LICENSE,
                    "upstream_split": upstream_split,
                    "upstream_index": index,
                    "content_hash": content_hash,
                    "split_group_id": group_id,
                    "split_policy_version": SPLIT_POLICY_VERSION,
                },
            )
            handle = handles[(source, split)]
            handle.write(json.dumps(_record_payload(record), sort_keys=True) + "\n")
            counts[source][split] += 1
            accepted += 1

        for handle in handles.values():
            handle.flush()
            handle.close()
        handles.clear()

        files: list[dict[str, object]] = []
        for path in sorted((temporary / "raw").rglob("*.jsonl")):
            files.append(
                {
                    "path": path.relative_to(temporary).as_posix(),
                    "byte_size": path.stat().st_size,
                    "sha256": sha256_path(path),
                }
            )

        manifest_without_hash: dict[str, object] = {
            "schema": "small-llm-sft-prepared-source",
            "schema_version": 1,
            "dataset_name": dataset_name,
            "revision": revision,
            "license_id": SMOL_SMOLTALK_LICENSE,
            "upstream_split": upstream_split,
            "allowed_sources": sorted(allowed),
            "split_policy": policy.as_dict(),
            "counts": counts,
            "files": files,
            "audit": {
                "seen": seen,
                "accepted": accepted,
                "exact_duplicates_removed": duplicates,
                "behavior_suite_contamination_removed": contaminated,
                "malformed": malformed,
                "unsupported_source": unsupported_source,
            },
        }
        manifest = {**manifest_without_hash, "manifest_sha256": canonical_hash(manifest_without_hash)}
        (temporary / "prepared-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.rename(root)
        return manifest
    except BaseException:
        for handle in handles.values():
            try:
                handle.close()
            except Exception:
                pass
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _split_target_tokens(train_target_tokens: int, split: Split) -> int:
    if train_target_tokens <= 0:
        raise ValueError("train_target_tokens must be positive")
    if split == "train":
        return train_target_tokens
    return max(1, math.floor(train_target_tokens * 0.025 / 0.95))


def _empty_replay() -> Iterator[TokenizedSFTRecord]:
    if False:  # pragma: no cover
        yield TokenizedSFTRecord("", "", "train", (), ())


def build_bundle(
    prepared_dir: Path | str,
    *,
    replay_root: Path | str,
    output_dir: Path | str,
    train_target_tokens: int,
    optimizer_target_tokens: int = 32_768,
    context_length: int = 2_048,
    maximum_assistant_tokens: int = 512,
    instruction_share: float = 0.85,
    replay_share: float = 0.15,
    instruction_source_shares: Mapping[str, float] = DEFAULT_INSTRUCTION_SOURCE_SHARES,
    seed: int = 17,
) -> dict[str, object]:
    """Build immutable train/validation/test SFT datasets and one bundle manifest."""

    prepared = Path(prepared_dir)
    prepared_manifest_path = prepared / "prepared-manifest.json"
    try:
        prepared_manifest = json.loads(prepared_manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError("prepared SFT source manifest is missing or invalid") from error
    if not isinstance(prepared_manifest, Mapping):
        raise RuntimeError("prepared SFT source manifest must be an object")
    supplied = prepared_manifest.get("manifest_sha256")
    without_hash = {key: value for key, value in prepared_manifest.items() if key != "manifest_sha256"}
    if supplied != canonical_hash(without_hash):
        raise RuntimeError("prepared SFT source manifest self-hash mismatch")
    raw_files = prepared_manifest.get("files")
    if not isinstance(raw_files, list):
        raise RuntimeError("prepared SFT source manifest has no file identities")
    for item in raw_files:
        if not isinstance(item, Mapping):
            raise RuntimeError("prepared SFT source file identity is malformed")
        relative = item.get("path")
        digest = item.get("sha256")
        byte_size = item.get("byte_size")
        if (
            not isinstance(relative, str)
            or not isinstance(digest, str)
            or isinstance(byte_size, bool)
            or not isinstance(byte_size, int)
        ):
            raise RuntimeError("prepared SFT source file identity is malformed")
        path = prepared / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"prepared SFT source file is missing or unsafe: {relative}")
        if path.stat().st_size != byte_size or sha256_path(path) != digest:
            raise RuntimeError(f"prepared SFT source file identity mismatch: {relative}")

    replay_manifest_path = Path(replay_root) / "manifest.json"
    if not replay_manifest_path.is_file() or replay_manifest_path.is_symlink():
        raise RuntimeError("replay root has no safe immutable manifest.json")
    replay_manifest_sha256 = sha256_path(replay_manifest_path)

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to replace existing SFT bundle: {output}")
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    split_results: dict[str, object] = {}
    try:
        for split in ("train", "validation", "test"):
            typed_split: Split = split  # type: ignore[assignment]
            config = SFTDataConfig(
                target_loss_tokens=_split_target_tokens(train_target_tokens, typed_split),
                optimizer_target_tokens=optimizer_target_tokens,
                context_length=context_length,
                maximum_assistant_tokens=maximum_assistant_tokens,
                instruction_share=instruction_share if split == "train" else 1.0,
                replay_share=replay_share if split == "train" else 0.0,
                instruction_source_shares=dict(instruction_source_shares),
                seed=seed,
            )
            source_paths = {
                source: prepared / "raw" / split / f"{source}.jsonl"
                for source in instruction_source_shares
            }
            missing = [str(path) for path in source_paths.values() if not path.is_file()]
            if missing:
                raise RuntimeError(f"prepared SFT split is missing source files: {missing}")
            sources = {
                source: JsonlConversationSource(path, expected_source=source, default_source=source)
                for source, path in source_paths.items()
            }
            replay = (
                iter_schema_v2_replay(replay_root, split="train", context_length=context_length)
                if split == "train"
                else _empty_replay()
            )
            split_root = temporary / split
            result = SFTDatasetBuilder(config, encoder=TiktokenGPT2Encoder()).build(
                instruction_sources=sources,
                replay_source=replay,
                output_dir=split_root,
            )
            split_results[split] = {
                "path": split,
                "manifest_sha256": result["manifest"]["manifest_sha256"],
                "loss_bearing_target_tokens": result["manifest"]["totals"]["loss_bearing_target_tokens"],
                "build_report_sha256": result["report"]["report_sha256"],
            }

        (temporary / "source-manifest.json").write_text(
            json.dumps(dict(prepared_manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        bundle_without_hash = {
            "schema": "small-llm-sft-bundle",
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "prepared_source_manifest_sha256": supplied,
            "prepared_source": {
                "dataset_name": prepared_manifest.get("dataset_name"),
                "revision": prepared_manifest.get("revision"),
                "license_id": prepared_manifest.get("license_id"),
                "split_policy": prepared_manifest.get("split_policy"),
            },
            "replay_manifest_sha256": replay_manifest_sha256,
            "train_target_tokens_requested": train_target_tokens,
            "optimizer_target_tokens": optimizer_target_tokens,
            "context_length": context_length,
            "maximum_assistant_tokens": maximum_assistant_tokens,
            "instruction_share": instruction_share,
            "replay_share": replay_share,
            "instruction_source_shares": dict(instruction_source_shares),
            "seed": seed,
            "splits": split_results,
        }
        manifest = {**bundle_without_hash, "manifest_sha256": canonical_hash(bundle_without_hash)}
        (temporary / "bundle-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def verify_bundle(root: Path | str) -> dict[str, object]:
    bundle = Path(root)
    try:
        manifest = json.loads((bundle / "bundle-manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError("SFT bundle manifest is missing or invalid") from error
    if not isinstance(manifest, Mapping):
        raise RuntimeError("SFT bundle manifest must be an object")
    supplied = manifest.get("manifest_sha256")
    without_hash = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if supplied != canonical_hash(without_hash):
        raise RuntimeError("SFT bundle manifest self-hash mismatch")
    if manifest.get("schema") != "small-llm-sft-bundle":
        raise RuntimeError("unsupported SFT bundle schema")

    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        raise RuntimeError("SFT bundle has no split identities")
    verified: dict[str, object] = {}
    for split in ("train", "validation", "test"):
        row = splits.get(split)
        if not isinstance(row, Mapping):
            raise RuntimeError(f"SFT bundle has no {split} split")
        reader = SFTShardReader(bundle / str(row["path"]), split=split)
        total = sum(reader.block_target_counts)
        if reader.manifest_identity != row.get("manifest_sha256"):
            raise RuntimeError(f"SFT bundle {split} manifest identity mismatch")
        if total != row.get("loss_bearing_target_tokens"):
            raise RuntimeError(f"SFT bundle {split} target-token total mismatch")
        blocks = list(reader.iter_from_start())
        verified[split] = {
            "manifest_sha256": reader.manifest_identity,
            "blocks": len(blocks),
            "loss_bearing_target_tokens": total,
        }
    return {"status": "verified", "bundle_manifest_sha256": supplied, "splits": verified}


def sft_budget_from_parent(
    parent_consumed_tokens: int,
    *,
    numerator: int = 4,
    denominator: int = 100,
) -> int:
    if isinstance(parent_consumed_tokens, bool) or not isinstance(parent_consumed_tokens, int):
        raise ValueError("parent_consumed_tokens must be an integer")
    if parent_consumed_tokens <= 0:
        raise ValueError("parent_consumed_tokens must be positive")
    if numerator <= 0 or denominator <= 0 or numerator >= denominator:
        raise ValueError("SFT budget fraction must be in (0, 1)")
    return parent_consumed_tokens * numerator // denominator


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="prepare the pinned identity-safe SmolTalk source")
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--dataset-name", default=SMOL_SMOLTALK_DATASET)
    prepare.add_argument("--revision", default=SMOL_SMOLTALK_DATA_REVISION)
    prepare.add_argument("--seed", type=int, default=17)

    build = sub.add_parser("build", help="build a 4%-scaled immutable SFT bundle")
    build.add_argument("--prepared-dir", type=Path, required=True)
    build.add_argument("--replay-root", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--parent-consumed-tokens", type=_positive_int, required=True)
    build.add_argument("--optimizer-target-tokens", type=_positive_int, default=32_768)
    build.add_argument("--instruction-share", type=float, default=0.85)
    build.add_argument("--replay-share", type=float, default=0.15)
    build.add_argument("--seed", type=int, default=17)

    verify = sub.add_parser("verify", help="verify every split/shard in an SFT bundle")
    verify.add_argument("--dataset-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_smoltalk(
            args.output_dir,
            dataset_name=args.dataset_name,
            revision=args.revision,
            split_policy=IdentitySplitPolicy(seed=args.seed),
        )
    elif args.command == "build":
        parent_tokens = args.parent_consumed_tokens
        result = build_bundle(
            args.prepared_dir,
            replay_root=args.replay_root,
            output_dir=args.output_dir,
            train_target_tokens=sft_budget_from_parent(parent_tokens),
            optimizer_target_tokens=args.optimizer_target_tokens,
            instruction_share=args.instruction_share,
            replay_share=args.replay_share,
            seed=args.seed,
        )
        result = {"parent_consumed_tokens": parent_tokens, "sft_fraction": 0.04, "bundle": result}
    else:
        result = verify_bundle(args.dataset_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "IdentitySplitPolicy",
    "SMOL_SMOLTALK_DATASET",
    "SMOL_SMOLTALK_DATA_REVISION",
    "SMOL_SMOLTALK_LICENSE",
    "build_bundle",
    "conversation_content_hash",
    "conversation_group_id",
    "prepare_smoltalk",
    "sft_budget_from_parent",
    "verify_bundle",
]
