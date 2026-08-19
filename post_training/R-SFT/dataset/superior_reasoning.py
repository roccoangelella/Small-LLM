#!/usr/bin/env python3
"""Select a compact Stage-1 subset of Alibaba Superior Reasoning for R-SFT.

The first real Small-LLM reasoning corpus uses only the Stage-1 low-temperature
split and only ``science`` / ``instruction_following`` rows. Selection is by
GPT-2 token count of the parsed ``<think>...</think>`` body, with a default
25,000-row target and 30/70 science/instruction-following stratification.

This module intentionally prepares data only. The current production
``build_atomic.py`` still enforces the historical uniform Gemini R0 matrix and
must be generalized separately before a heterogeneous corpus is trained.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import argparse
import hashlib
import heapq
import json
from pathlib import Path
import random
import tempfile
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence, TextIO

DATASET_ID = "Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b"
DATASET_CONFIG = "stage1"
DATASET_SPLIT = "train"
ELIGIBLE_DOMAINS = ("science", "instruction_following")
DEFAULT_TOTAL = 25_000
DEFAULT_SCIENCE_SHARE = 0.30
DEFAULT_SEED = 17
SOURCE_SKILLS = {
    "science": "SR_SCIENCE",
    "instruction_following": "SR_INSTRUCTION_FOLLOWING",
}
SOURCE_DIFFICULTY = "shortest"


@dataclass(frozen=True, slots=True)
class ParsedOutput:
    reasoning: str
    answer: str


@dataclass(frozen=True, slots=True)
class CandidateRef:
    offset: int
    source_index: int
    source_id: str
    domain: str
    reasoning_token_count: int


@dataclass(frozen=True, slots=True)
class SelectionReport:
    source_rows: int
    domain_counts: dict[str, int]
    usable_counts: dict[str, int]
    selected_counts: dict[str, int]
    rejected_output_count: int
    duplicate_input_count: int
    total_selected: int
    requested_total: int
    requested_science_share: float
    realized_science_share: float
    selected_min_reasoning_tokens: int | None
    selected_max_reasoning_tokens: int | None

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset_id": DATASET_ID,
            "dataset_config": DATASET_CONFIG,
            "dataset_split": DATASET_SPLIT,
            "source_rows": self.source_rows,
            "domain_counts": dict(sorted(self.domain_counts.items())),
            "usable_counts": dict(sorted(self.usable_counts.items())),
            "selected_counts": dict(sorted(self.selected_counts.items())),
            "rejected_output_count": self.rejected_output_count,
            "duplicate_input_count": self.duplicate_input_count,
            "total_selected": self.total_selected,
            "requested_total": self.requested_total,
            "requested_science_share": self.requested_science_share,
            "realized_science_share": self.realized_science_share,
            "selected_min_reasoning_tokens": self.selected_min_reasoning_tokens,
            "selected_max_reasoning_tokens": self.selected_max_reasoning_tokens,
        }


def parse_teacher_output(output: str) -> ParsedOutput:
    """Split one Superior response into reasoning and final answer.

    Superior Stage 1 examples are distilled with a ``<think>`` block. We fail
    closed when its closing marker or final answer is missing instead of
    silently teaching malformed serialization.
    """

    if not isinstance(output, str) or not output.strip():
        raise ValueError("Superior output must be non-empty text")
    text = output.strip()
    if not text.startswith("<think>"):
        raise ValueError("Superior output must start with <think>")
    close = text.find("</think>", len("<think>"))
    if close < 0:
        raise ValueError("Superior output is missing </think>")

    reasoning = text[len("<think>") : close].strip()
    answer = text[close + len("</think>") :].strip()
    for tag in ("<answer>", "<final>"):
        if answer.startswith(tag):
            answer = answer[len(tag) :].lstrip()
            closing = "</" + tag[1:]
            if answer.endswith(closing):
                answer = answer[: -len(closing)].rstrip()
            break

    if not reasoning:
        raise ValueError("Superior output has an empty reasoning body")
    if not answer:
        raise ValueError("Superior output has no final answer after </think>")
    return ParsedOutput(reasoning=reasoning, answer=answer)


def _normalized_input_hash(text: str) -> str:
    normalized = " ".join(text.split()).casefold().encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def _row_text(row: Mapping[str, Any], field: str, *, index: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Superior row {index} field {field!r} must be non-empty text")
    return value.strip()


def _default_token_counter() -> Callable[[str], int]:
    try:
        import tiktoken
    except ImportError as error:  # pragma: no cover - environment dependency
        raise RuntimeError(
            "tiktoken is required to rank Superior reasoning traces by the project GPT-2 tokenizer"
        ) from error
    encoding = tiktoken.get_encoding("gpt2")

    def count(text: str) -> int:
        return len(encoding.encode(text, allowed_special=set(), disallowed_special=()))

    return count


def iter_stage1_rows() -> Iterator[Mapping[str, Any]]:
    """Stream the public Stage-1 split without materializing its ~4.6 GB JSONL."""

    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - environment dependency
        raise RuntimeError(
            "huggingface datasets is required; install it before building Superior Reasoning data"
        ) from error

    dataset = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split=DATASET_SPLIT,
        streaming=True,
    )
    yield from dataset


def _spool_candidate(
    handle: TextIO,
    *,
    source_index: int,
    source_id: str,
    domain: str,
    problem: str,
    reasoning: str,
    answer: str,
    reasoning_token_count: int,
) -> int:
    offset = handle.tell()
    payload = {
        "source_index": source_index,
        "source_id": source_id,
        "domain": domain,
        "problem": problem,
        "reasoning": reasoning,
        "answer": answer,
        "reasoning_token_count": reasoning_token_count,
    }
    handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    handle.write("\n")
    return offset


def _push_shortest(
    heap: list[tuple[int, int, CandidateRef]],
    candidate: CandidateRef,
    *,
    capacity: int,
) -> None:
    """Keep the ``capacity`` lowest (token_count, source_index) candidates."""

    entry = (-candidate.reasoning_token_count, -candidate.source_index, candidate)
    if len(heap) < capacity:
        heapq.heappush(heap, entry)
        return
    worst_tokens = -heap[0][0]
    worst_index = -heap[0][1]
    if (candidate.reasoning_token_count, candidate.source_index) < (
        worst_tokens,
        worst_index,
    ):
        heapq.heapreplace(heap, entry)


def _allocate_counts(
    *,
    total: int,
    science_share: float,
    usable_counts: Mapping[str, int],
) -> dict[str, int]:
    if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
        raise ValueError("total must be a positive integer")
    if not 0.0 <= science_share <= 1.0:
        raise ValueError("science_share must be in [0, 1]")

    science_target = round(total * science_share)
    instruction_target = total - science_target
    counts = {
        "science": min(science_target, int(usable_counts.get("science", 0))),
        "instruction_following": min(
            instruction_target, int(usable_counts.get("instruction_following", 0))
        ),
    }
    deficit = total - sum(counts.values())
    if deficit:
        # Backfill first from the domain with the largest remaining pool. Ties
        # prefer instruction-following because it is the requested majority.
        domains = sorted(
            ELIGIBLE_DOMAINS,
            key=lambda domain: (
                int(usable_counts.get(domain, 0)) - counts[domain],
                domain == "instruction_following",
            ),
            reverse=True,
        )
        for domain in domains:
            available = int(usable_counts.get(domain, 0)) - counts[domain]
            take = min(deficit, max(0, available))
            counts[domain] += take
            deficit -= take
            if deficit == 0:
                break
    if deficit:
        raise RuntimeError(
            f"only {total - deficit} usable science/instruction rows are available; requested {total}"
        )
    return counts


def _read_spooled(handle: TextIO, offset: int) -> dict[str, Any]:
    handle.seek(offset)
    line = handle.readline()
    if not line:
        raise RuntimeError(f"candidate spool offset {offset} is unreadable")
    payload = json.loads(line)
    if not isinstance(payload, dict):
        raise RuntimeError("candidate spool contains a non-object row")
    return payload


def _selection_order_key(candidate: CandidateRef, *, seed: int) -> str:
    payload = f"{seed}\x1f{candidate.source_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_shortest(
    rows: Iterable[Mapping[str, Any]],
    *,
    total: int = DEFAULT_TOTAL,
    science_share: float = DEFAULT_SCIENCE_SHARE,
    seed: int = DEFAULT_SEED,
    token_counter: Callable[[str], int] | None = None,
    spool_dir: str | Path | None = None,
) -> tuple[list[dict[str, Any]], SelectionReport]:
    """Scan once, count domains, and select the shortest stratified traces.

    Candidate text is spooled to disk while only small heap references stay in
    memory, so the 4.6 GB Stage-1 source does not become a multi-GB Python list.
    Exact normalized duplicate prompts are dropped before ranking.
    """

    if token_counter is None:
        token_counter = _default_token_counter()
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    domain_counts: Counter[str] = Counter()
    usable_counts: Counter[str] = Counter()
    heaps: dict[str, list[tuple[int, int, CandidateRef]]] = {
        domain: [] for domain in ELIGIBLE_DOMAINS
    }
    seen_inputs: set[str] = set()
    rejected_output_count = 0
    duplicate_input_count = 0
    source_rows = 0

    directory = None if spool_dir is None else str(Path(spool_dir))
    with tempfile.NamedTemporaryFile(
        mode="w+",
        encoding="utf-8",
        prefix="small-llm-superior-",
        suffix=".jsonl",
        dir=directory,
    ) as spool:
        for source_index, row in enumerate(rows):
            source_rows += 1
            if not isinstance(row, Mapping):
                raise ValueError(f"Superior row {source_index} must be an object")
            domain_raw = row.get("domain")
            domain = domain_raw.strip() if isinstance(domain_raw, str) else "<invalid>"
            domain_counts[domain] += 1
            if domain not in ELIGIBLE_DOMAINS:
                continue

            problem = _row_text(row, "input", index=source_index)
            source_id = _row_text(row, "uuid", index=source_index)
            output = _row_text(row, "output", index=source_index)
            try:
                parsed = parse_teacher_output(output)
            except ValueError:
                rejected_output_count += 1
                continue

            input_hash = _normalized_input_hash(problem)
            if input_hash in seen_inputs:
                duplicate_input_count += 1
                continue
            seen_inputs.add(input_hash)

            reasoning_token_count = int(token_counter(parsed.reasoning))
            if reasoning_token_count <= 0:
                raise ValueError("token_counter must return a positive count")

            usable_counts[domain] += 1
            offset = _spool_candidate(
                spool,
                source_index=source_index,
                source_id=source_id,
                domain=domain,
                problem=problem,
                reasoning=parsed.reasoning,
                answer=parsed.answer,
                reasoning_token_count=reasoning_token_count,
            )
            candidate = CandidateRef(
                offset=offset,
                source_index=source_index,
                source_id=source_id,
                domain=domain,
                reasoning_token_count=reasoning_token_count,
            )
            _push_shortest(heaps[domain], candidate, capacity=total)

        selected_counts = _allocate_counts(
            total=total,
            science_share=science_share,
            usable_counts=usable_counts,
        )
        selected_refs: list[CandidateRef] = []
        for domain in ELIGIBLE_DOMAINS:
            ranked = sorted(
                (entry[2] for entry in heaps[domain]),
                key=lambda candidate: (
                    candidate.reasoning_token_count,
                    candidate.source_index,
                ),
            )
            selected_refs.extend(ranked[: selected_counts[domain]])

        # Deterministically interleave domains without making selection depend
        # on Python's random implementation or source ordering.
        selected_refs.sort(key=lambda candidate: _selection_order_key(candidate, seed=seed))
        selected = [_read_spooled(spool, candidate.offset) for candidate in selected_refs]

    token_counts = [candidate.reasoning_token_count for candidate in selected_refs]
    science_selected = selected_counts["science"]
    report = SelectionReport(
        source_rows=source_rows,
        domain_counts=dict(domain_counts),
        usable_counts=dict(usable_counts),
        selected_counts=selected_counts,
        rejected_output_count=rejected_output_count,
        duplicate_input_count=duplicate_input_count,
        total_selected=len(selected),
        requested_total=total,
        requested_science_share=science_share,
        realized_science_share=science_selected / len(selected) if selected else 0.0,
        selected_min_reasoning_tokens=min(token_counts) if token_counts else None,
        selected_max_reasoning_tokens=max(token_counts) if token_counts else None,
    )
    return selected, report


def to_rsft_mapping(record: Mapping[str, Any]) -> dict[str, str]:
    domain = str(record["domain"])
    try:
        skill = SOURCE_SKILLS[domain]
    except KeyError as error:
        raise ValueError(f"unsupported Superior domain {domain!r}") from error
    return {
        "skill": skill,
        "difficulty": SOURCE_DIFFICULTY,
        "problem": str(record["problem"]).strip(),
        "reasoning": str(record["reasoning"]).strip(),
        "answer": str(record["answer"]).strip(),
    }


def _read_gemini_jsonl(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    expected = {"skill", "difficulty", "problem", "reasoning", "answer"}
    records: list[dict[str, str]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Gemini JSONL contains a blank line at {line_number}")
            raw = json.loads(line)
            if not isinstance(raw, Mapping) or set(raw) != expected:
                raise ValueError(
                    f"Gemini JSONL line {line_number} must contain exactly {sorted(expected)}"
                )
            record: dict[str, str] = {}
            for field in sorted(expected):
                value = raw[field]
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"Gemini JSONL line {line_number} field {field!r} must be non-empty text"
                    )
                record[field] = value.strip()
            records.append(record)
    if not records:
        raise ValueError("Gemini JSONL contains no records")
    return records


def write_combined_jsonl(
    superior_records: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    *,
    gemini_jsonl: str | Path | None = None,
    seed: int = DEFAULT_SEED,
) -> Path:
    """Write Superior rows plus the existing Gemini R0 rows in deterministic order."""

    combined = [to_rsft_mapping(record) for record in superior_records]
    if gemini_jsonl is not None:
        combined.extend(_read_gemini_jsonl(gemini_jsonl))
    rng = random.Random(seed)
    rng.shuffle(combined)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in combined:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    return destination


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset(
    output_jsonl: str | Path,
    *,
    gemini_jsonl: str | Path | None = None,
    total: int = DEFAULT_TOTAL,
    science_share: float = DEFAULT_SCIENCE_SHARE,
    seed: int = DEFAULT_SEED,
    spool_dir: str | Path | None = None,
) -> dict[str, object]:
    selected, report = select_shortest(
        iter_stage1_rows(),
        total=total,
        science_share=science_share,
        seed=seed,
        spool_dir=spool_dir,
    )
    destination = write_combined_jsonl(
        selected,
        output_jsonl,
        gemini_jsonl=gemini_jsonl,
        seed=seed,
    )
    gemini_rows = len(_read_gemini_jsonl(gemini_jsonl)) if gemini_jsonl is not None else 0
    result = report.as_dict()
    result.update(
        {
            "schema": "small-llm-superior-reasoning-selection-v1",
            "gemini_rows": gemini_rows,
            "combined_rows": len(selected) + gemini_rows,
            "output_jsonl": str(destination),
            "output_sha256": _sha256(destination),
            "seed": seed,
            "ranking_metric": "gpt2_reasoning_token_count",
            "eligible_domains": list(ELIGIBLE_DOMAINS),
        }
    )
    manifest = destination.with_suffix(destination.suffix + ".manifest.json")
    manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["manifest"] = str(manifest)
    return result


def scan_dataset() -> dict[str, object]:
    """Return exact Stage-1 domain counts without tokenizing or selecting rows."""

    counts: Counter[str] = Counter()
    rows = 0
    for index, row in enumerate(iter_stage1_rows()):
        rows += 1
        if not isinstance(row, Mapping):
            raise ValueError(f"Superior row {index} must be an object")
        raw = row.get("domain")
        domain = raw.strip() if isinstance(raw, str) else "<invalid>"
        counts[domain] += 1
    return {
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_split": DATASET_SPLIT,
        "rows": rows,
        "domain_counts": dict(sorted(counts.items())),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("scan", help="print exact Stage-1 domain counts")

    build = subparsers.add_parser("build", help="select 25k shortest rows and write R-SFT JSONL")
    build.add_argument("--output-jsonl", type=Path, required=True)
    build.add_argument("--gemini-jsonl", type=Path)
    build.add_argument("--total", type=int, default=DEFAULT_TOTAL)
    build.add_argument("--science-share", type=float, default=DEFAULT_SCIENCE_SHARE)
    build.add_argument("--seed", type=int, default=DEFAULT_SEED)
    build.add_argument("--spool-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "scan":
        print(json.dumps(scan_dataset(), indent=2, sort_keys=True))
        return 0
    result = build_dataset(
        args.output_jsonl,
        gemini_jsonl=args.gemini_jsonl,
        total=args.total,
        science_share=args.science_share,
        seed=args.seed,
        spool_dir=args.spool_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
