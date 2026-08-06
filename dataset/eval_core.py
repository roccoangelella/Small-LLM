"""Build and verify the permanent stratified ``eval_core_v1`` corpus.

The builder reads only documents assigned to the repository's existing
validation partition.  It writes two nested suites:

* ``fast``: at least 32 documents and 16,384 scored targets per retained cluster;
* ``full``: at least 256 documents and 131,072 scored targets per retained cluster.

Each sequence is stored as 2,049 little-endian uint16 token IDs.  The companion
JSONL row records its source document, cluster, and number of valid targets so
the evaluator can ignore padding and bootstrap by independent document.
"""
from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Iterable, Iterator, Mapping, Sequence

from dataset import config
from dataset.src.bytesource import HttpRangeReader, SourceFile, list_source_files
from dataset.src.records import (
    ParsedRecord,
    iter_owned_records,
    record_identity_str,
    validate_record,
)
from dataset.src.split import is_validation
from dataset.src.workplan import build_work_plan

EVAL_NAME = "eval_core_v1"
SCHEMA_VERSION = 1
CONTEXT_LENGTH = 2_048
STORED_TOKENS = CONTEXT_LENGTH + 1
FAST_DOCUMENTS_PER_CLUSTER = 32
FAST_TARGETS_PER_CLUSTER = 16_384
FULL_DOCUMENTS_PER_CLUSTER = 256
FULL_TARGETS_PER_CLUSTER = 131_072
ACCEPTED_CLUSTERS = tuple(sorted(config.ACCEPTED_CLUSTER_IDS))

# Exact source-token totals from the approved full-corpus calibration.
MIXTURE_SOURCE_TOKENS: dict[int, int] = {
    1: 3_063_924_776,
    2: 4_582_654_670,
    3: 5_771_522_873,
    4: 12_797_349_739,
    5: 6_563_519_102,
    6: 70_958_523_234,
    7: 64_041_261_564,
    8: 3_818_960_760,
    9: 2_978_940_628,
    10: 26_975_813_961,
    12: 78_310_785_713,
    13: 2_561_634_721,
    14: 834_424_339,
    15: 804_163_375,
    16: 27_851_136_242,
    17: 26_785_891_472,
    18: 7_392_477_294,
    19: 3_836_811_747,
    20: 1_862_658_535,
}
MIXTURE_WEIGHTS_SHA256 = (
    "76e82e22760adcac59c7294fe9bac11358f5a8b7a26035aae64c3f2e6fa1acb7"
)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_file(path: Path, *, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def document_windows(
    tokens: Sequence[int],
    *,
    context_length: int = CONTEXT_LENGTH,
    eod_token_id: int = config.EOD_TOKEN_ID,
) -> tuple[tuple[tuple[int, ...], int], ...]:
    """Return padded context+1 sequences and valid-target counts for one document."""
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    normalized = [int(token) for token in tokens]
    if not normalized:
        return ()
    if any(token < config.TOKEN_MIN or token > config.TOKEN_MAX for token in normalized):
        raise ValueError("document contains a token outside the semantic vocabulary")
    if normalized[-1] != eod_token_id:
        normalized.append(eod_token_id)
    if len(normalized) < 2:
        return ()

    stored = context_length + 1
    rows: list[tuple[tuple[int, ...], int]] = []
    for start in range(0, len(normalized) - 1, context_length):
        valid_targets = min(context_length, len(normalized) - 1 - start)
        if valid_targets <= 0:
            break
        sequence = normalized[start : start + stored]
        if len(sequence) < stored:
            sequence.extend([eod_token_id] * (stored - len(sequence)))
        rows.append((tuple(sequence), valid_targets))
    return tuple(rows)


class _SuiteWriter:
    def __init__(self, root: Path, name: str) -> None:
        self.name = name
        self.data_path = root / f"{name}.bin"
        self.records_path = root / f"{name}.records.jsonl"
        self._data = self.data_path.open("wb")
        self._records = self.records_path.open("w", encoding="utf-8", newline="\n")
        self.sequence_count = 0
        self.document_ids: set[str] = set()
        self.target_tokens = 0
        self.per_cluster = {
            cluster: {"documents": 0, "target_tokens": 0, "sequences": 0}
            for cluster in ACCEPTED_CLUSTERS
        }

    def add_document(
        self,
        *,
        document_id: str,
        cluster_id: int,
        filename: str,
        record_start: int,
        windows: Sequence[tuple[Sequence[int], int]],
    ) -> None:
        if document_id in self.document_ids:
            raise RuntimeError(f"duplicate eval document identity: {document_id}")
        self.document_ids.add(document_id)
        document_targets = 0
        for document_window, valid_targets in windows:
            if len(document_window) != STORED_TOKENS:
                raise ValueError("eval window has the wrong stored-token length")
            payload = array("H", document_window)
            if sys.byteorder != "little":
                payload.byteswap()
            self._data.write(payload.tobytes())
            row = {
                "sequence_index": self.sequence_count,
                "document_id": document_id,
                "cluster_id": cluster_id,
                "filename": filename,
                "record_start": record_start,
                "valid_targets": int(valid_targets),
            }
            self._records.write(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            )
            self.sequence_count += 1
            document_targets += int(valid_targets)
            self.per_cluster[cluster_id]["sequences"] += 1

        self.target_tokens += document_targets
        self.per_cluster[cluster_id]["documents"] += 1
        self.per_cluster[cluster_id]["target_tokens"] += document_targets

    def close(self) -> None:
        self._data.flush()
        os.fsync(self._data.fileno())
        self._data.close()
        self._records.flush()
        os.fsync(self._records.fileno())
        self._records.close()

    def summary(self) -> dict[str, object]:
        return {
            "data_file": self.data_path.name,
            "records_file": self.records_path.name,
            "data_sha256": sha256_file(self.data_path),
            "records_sha256": sha256_file(self.records_path),
            "sequence_count": self.sequence_count,
            "document_count": len(self.document_ids),
            "target_token_count": self.target_tokens,
            "per_cluster": {
                str(cluster): dict(self.per_cluster[cluster])
                for cluster in ACCEPTED_CLUSTERS
            },
        }


def _needs(
    writer: _SuiteWriter, cluster_id: int, *, documents: int, targets: int
) -> bool:
    row = writer.per_cluster[cluster_id]
    return row["documents"] < documents or row["target_tokens"] < targets


def _complete(writer: _SuiteWriter, *, documents: int, targets: int) -> bool:
    return all(
        not _needs(writer, cluster, documents=documents, targets=targets)
        for cluster in ACCEPTED_CLUSTERS
    )


def _iter_source_records(
    source_files: Sequence[SourceFile],
    *,
    max_work_items: int | None = None,
) -> Iterator[tuple[str, ParsedRecord]]:
    plan = build_work_plan(
        list(source_files),
        region_bytes=config.REGION_BYTES,
        seed=config.SELECTION_SEED,
        repository=config.DATASET_REPOSITORY,
        revision=config.DATASET_REVISION,
    )
    by_name = {source.path: source for source in source_files}
    readers: dict[str, HttpRangeReader] = {}
    for ordinal, item in enumerate(plan.work_items):
        if max_work_items is not None and ordinal >= max_work_items:
            break
        source = by_name[item.filename]
        reader = readers.get(item.filename)
        if reader is None:
            reader = HttpRangeReader(
                source, config.DATASET_REPOSITORY, config.DATASET_REVISION
            )
            readers[item.filename] = reader
        for record in iter_owned_records(item, reader):
            yield item.filename, record


def build_eval_core(
    output_dir: Path,
    *,
    max_work_items: int | None = None,
    record_stream: Iterable[tuple[str, ParsedRecord]] | None = None,
    validation_probability: float = config.VALIDATION_PROBABILITY,
    fast_documents_per_cluster: int = FAST_DOCUMENTS_PER_CLUSTER,
    fast_targets_per_cluster: int = FAST_TARGETS_PER_CLUSTER,
    full_documents_per_cluster: int = FULL_DOCUMENTS_PER_CLUSTER,
    full_targets_per_cluster: int = FULL_TARGETS_PER_CLUSTER,
) -> dict[str, object]:
    """Build ``eval_core_v1`` atomically.

    ``record_stream`` and the quota arguments are injection points for offline
    tests.  The production CLI exposes only the frozen defaults.
    """
    for name, value in {
        "fast_documents_per_cluster": fast_documents_per_cluster,
        "fast_targets_per_cluster": fast_targets_per_cluster,
        "full_documents_per_cluster": full_documents_per_cluster,
        "full_targets_per_cluster": full_targets_per_cluster,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if fast_documents_per_cluster > full_documents_per_cluster:
        raise ValueError("fast document floor cannot exceed full")
    if fast_targets_per_cluster > full_targets_per_cluster:
        raise ValueError("fast target floor cannot exceed full")
    if not 0.0 <= validation_probability <= 1.0:
        raise ValueError("validation_probability must be in [0, 1]")

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to replace existing eval directory: {output_dir}")
    temporary = output_dir.with_name(f".{output_dir.name}.partial-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    fast = _SuiteWriter(temporary, "fast")
    full = _SuiteWriter(temporary, "full")
    selected_full: set[str] = set()
    scanned_records = 0
    validation_records = 0
    try:
        if record_stream is None:
            sources = list_source_files(
                config.DATASET_REPOSITORY, config.DATASET_REVISION
            )
            stream = _iter_source_records(sources, max_work_items=max_work_items)
        else:
            sources = ()
            stream = iter(record_stream)

        for filename, record in stream:
            scanned_records += 1
            result = validate_record(record)
            if (
                not result.valid
                or result.cluster_id not in config.ACCEPTED_CLUSTER_IDS
                or result.tokens is None
            ):
                continue
            if not is_validation(
                seed=config.SELECTION_SEED,
                revision=config.DATASET_REVISION,
                filename=filename,
                record_start=record.record_start,
                probability=validation_probability,
            ):
                continue
            validation_records += 1
            cluster_id = int(result.cluster_id)
            if not _needs(
                full,
                cluster_id,
                documents=full_documents_per_cluster,
                targets=full_targets_per_cluster,
            ):
                if _complete(
                    full,
                    documents=full_documents_per_cluster,
                    targets=full_targets_per_cluster,
                ):
                    break
                continue
            windows = document_windows(result.tokens)
            if not windows:
                continue
            document_id = record_identity_str(
                config.DATASET_REVISION, filename, record.record_start
            )
            if document_id in selected_full:
                raise RuntimeError(f"source record selected twice: {document_id}")
            selected_full.add(document_id)
            full.add_document(
                document_id=document_id,
                cluster_id=cluster_id,
                filename=filename,
                record_start=record.record_start,
                windows=windows,
            )
            if _needs(
                fast,
                cluster_id,
                documents=fast_documents_per_cluster,
                targets=fast_targets_per_cluster,
            ):
                fast.add_document(
                    document_id=document_id,
                    cluster_id=cluster_id,
                    filename=filename,
                    record_start=record.record_start,
                    windows=windows,
                )
            if validation_records % 1_000 == 0:
                print(
                    f"validation_docs={validation_records:,} "
                    f"full_targets={full.target_tokens:,} "
                    f"full_docs={len(full.document_ids):,}",
                    flush=True,
                )

        if not _complete(
            full,
            documents=full_documents_per_cluster,
            targets=full_targets_per_cluster,
        ):
            missing = {
                str(cluster): full.per_cluster[cluster]
                for cluster in ACCEPTED_CLUSTERS
                if _needs(
                    full,
                    cluster,
                    documents=full_documents_per_cluster,
                    targets=full_targets_per_cluster,
                )
            }
            raise RuntimeError(
                "source scan ended before the frozen full quotas were met: "
                + json.dumps(missing, sort_keys=True)
            )
        if not _complete(
            fast,
            documents=fast_documents_per_cluster,
            targets=fast_targets_per_cluster,
        ):
            raise RuntimeError("full quotas passed but nested fast quotas did not")
    except Exception:
        fast.close()
        full.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    fast.close()
    full.close()

    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "name": EVAL_NAME,
        "dataset": config.DATASET_REPOSITORY,
        "revision": config.DATASET_REVISION,
        "tokenizer": config.TOKENIZER_ID,
        "semantic_vocab_size": config.VOCAB_SIZE,
        "eod_token_id": config.EOD_TOKEN_ID,
        "context_length": CONTEXT_LENGTH,
        "stored_tokens_per_sequence": STORED_TOKENS,
        "dtype": "uint16-le",
        "split": {
            "seed": config.SELECTION_SEED,
            "hash_version": config.SPLIT_HASH_VERSION,
            "validation_probability": validation_probability,
        },
        "accepted_clusters": list(ACCEPTED_CLUSTERS),
        "mixture_source_tokens": {
            str(cluster): MIXTURE_SOURCE_TOKENS[cluster]
            for cluster in ACCEPTED_CLUSTERS
        },
        "mixture_weights_sha256": MIXTURE_WEIGHTS_SHA256,
        "selection_order": "frozen hash-shuffled source work plan",
        "scanned_records": scanned_records,
        "validation_records_seen": validation_records,
        "source_file_count": len(sources),
        "minimums": {
            "fast": {
                "documents_per_cluster": fast_documents_per_cluster,
                "target_tokens_per_cluster": fast_targets_per_cluster,
            },
            "full": {
                "documents_per_cluster": full_documents_per_cluster,
                "target_tokens_per_cluster": full_targets_per_cluster,
            },
        },
        "suites": {"fast": fast.summary(), "full": full.summary()},
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    manifest_path = temporary / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with manifest_path.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, output_dir)
    return manifest


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_records(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except ValueError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(row)
    return rows


def verify_eval_core(
    eval_dir: Path,
    *,
    enforce_frozen_minimums: bool = True,
) -> dict[str, object]:
    eval_dir = eval_dir.resolve()
    manifest = _read_json(eval_dir / "manifest.json")
    stored_self_hash = manifest.get("manifest_sha256")
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    computed_self_hash = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if stored_self_hash != computed_self_hash:
        raise ValueError("eval manifest self-hash mismatch")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("name") != EVAL_NAME:
        raise ValueError("unsupported eval manifest identity")
    if manifest.get("revision") != config.DATASET_REVISION:
        raise ValueError("eval source revision mismatch")
    if manifest.get("context_length") != CONTEXT_LENGTH:
        raise ValueError("eval context length mismatch")
    if manifest.get("accepted_clusters") != list(ACCEPTED_CLUSTERS):
        raise ValueError("eval cluster set mismatch")

    suites = manifest.get("suites")
    if not isinstance(suites, Mapping):
        raise ValueError("manifest has no suites object")
    document_sets: dict[str, set[str]] = {}
    for suite_name in ("fast", "full"):
        suite = suites.get(suite_name)
        if not isinstance(suite, Mapping):
            raise ValueError(f"manifest has no {suite_name} suite")
        data_path = eval_dir / str(suite.get("data_file", ""))
        records_path = eval_dir / str(suite.get("records_file", ""))
        if not data_path.is_file() or not records_path.is_file():
            raise ValueError(f"{suite_name} suite files are missing")
        if sha256_file(data_path) != suite.get("data_sha256"):
            raise ValueError(f"{suite_name} data hash mismatch")
        if sha256_file(records_path) != suite.get("records_sha256"):
            raise ValueError(f"{suite_name} records hash mismatch")
        sequence_count = suite.get("sequence_count")
        if not isinstance(sequence_count, int) or sequence_count <= 0:
            raise ValueError(f"{suite_name} sequence count is invalid")
        expected_bytes = sequence_count * STORED_TOKENS * 2
        if data_path.stat().st_size != expected_bytes:
            raise ValueError(f"{suite_name} binary size mismatch")
        records = _read_records(records_path)
        if len(records) != sequence_count:
            raise ValueError(f"{suite_name} records count mismatch")
        docs: set[str] = set()
        per_cluster = {
            cluster: {"documents": set(), "target_tokens": 0, "sequences": 0}
            for cluster in ACCEPTED_CLUSTERS
        }
        total_targets = 0
        for expected_index, row in enumerate(records):
            if row.get("sequence_index") != expected_index:
                raise ValueError(f"{suite_name} sequence indexes are not contiguous")
            document_id = row.get("document_id")
            cluster_id = row.get("cluster_id")
            valid_targets = row.get("valid_targets")
            if not isinstance(document_id, str) or not document_id:
                raise ValueError(f"{suite_name} has an invalid document ID")
            if cluster_id not in ACCEPTED_CLUSTERS:
                raise ValueError(f"{suite_name} has an invalid cluster ID")
            if (
                not isinstance(valid_targets, int)
                or valid_targets <= 0
                or valid_targets > CONTEXT_LENGTH
            ):
                raise ValueError(f"{suite_name} has an invalid target count")
            docs.add(document_id)
            row_cluster = per_cluster[int(cluster_id)]
            row_cluster["documents"].add(document_id)
            row_cluster["target_tokens"] += valid_targets
            row_cluster["sequences"] += 1
            total_targets += valid_targets
        if len(docs) != suite.get("document_count"):
            raise ValueError(f"{suite_name} document count mismatch")
        if total_targets != suite.get("target_token_count"):
            raise ValueError(f"{suite_name} target-token count mismatch")
        document_sets[suite_name] = docs

        if enforce_frozen_minimums:
            document_floor = (
                FAST_DOCUMENTS_PER_CLUSTER
                if suite_name == "fast"
                else FULL_DOCUMENTS_PER_CLUSTER
            )
            target_floor = (
                FAST_TARGETS_PER_CLUSTER
                if suite_name == "fast"
                else FULL_TARGETS_PER_CLUSTER
            )
            for cluster, observed in per_cluster.items():
                if len(observed["documents"]) < document_floor:
                    raise ValueError(
                        f"{suite_name} cluster {cluster} misses the document floor"
                    )
                if observed["target_tokens"] < target_floor:
                    raise ValueError(
                        f"{suite_name} cluster {cluster} misses the target floor"
                    )
    if not document_sets["fast"].issubset(document_sets["full"]):
        raise ValueError("fast documents are not a nested subset of full")
    return manifest


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or verify eval_core_v1")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="build the frozen fast and full suites")
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument(
        "--max-work-items",
        type=int,
        help="diagnostic scan bound; a production build normally leaves this unset",
    )
    verify = sub.add_parser("verify", help="verify hashes, geometry, quotas, and nesting")
    verify.add_argument("--eval-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    if args.command == "build":
        manifest = build_eval_core(args.output_dir, max_work_items=args.max_work_items)
        print(
            json.dumps(
                {
                    "status": "completed",
                    "output_dir": str(args.output_dir.resolve()),
                    "manifest_sha256": manifest["manifest_sha256"],
                    "suites": manifest["suites"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    manifest = verify_eval_core(args.eval_dir)
    print(
        json.dumps(
            {
                "status": "verified",
                "eval_dir": str(args.eval_dir.resolve()),
                "manifest_sha256": manifest["manifest_sha256"],
                "suites": manifest["suites"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
