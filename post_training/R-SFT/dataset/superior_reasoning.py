#!/usr/bin/env python3
"""Prepare Superior Reasoning data for the first large R-SFT run.

The accepted production policy is instruction-following only: Stage-1 ``science``,
``math``, and ``code`` rows are excluded, and instruction rows whose primary task
is mathematical computation or programming are conservatively removed. Every
kept Superior record must fit the model's real 2,048-token atomic R-SFT chat
serialization without truncation. The frozen 630-example Gemini logic corpus is
then merged as a small logic anchor.

The older 30/70 science/instruction selector remains below for reproducibility of
ADR 0102 and its local artifact, but it is no longer the production build path.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import argparse
import hashlib
import heapq
import io
import json
from pathlib import Path
import random
import re
import tempfile
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence, TextIO
from urllib.request import Request, urlopen

DATASET_ID = "Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b"
DATASET_CONFIG = "stage1"
DATASET_SPLIT = "train"
DATASET_REVISION = "main"
DATASET_FILENAME = "Superior-Reasoning-SFT-gpt-oss-120b-stage1-train-data.jsonl"
DATASET_URL = (
    f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/"
    f"{DATASET_FILENAME}?download=true"
)
ELIGIBLE_DOMAINS = ("science", "instruction_following")
DEFAULT_TOTAL = 25_000
DEFAULT_SCIENCE_SHARE = 0.30
DEFAULT_SEED = 17
DEFAULT_HTTP_TIMEOUT_SECONDS = 120.0
SOURCE_SKILLS = {
    "science": "SR_SCIENCE",
    "instruction_following": "SR_INSTRUCTION_FOLLOWING",
}
SOURCE_DIFFICULTY = "shortest"
PRODUCTION_DOMAIN = "instruction_following"
PRODUCTION_DIFFICULTY = "clean_fit"
PRODUCTION_CONTEXT_LENGTH = 2_048
PRODUCTION_FILTER_VERSION = "instruction-no-math-code-v1"
PRODUCTION_RESERVED_MARKERS = ("<think>", "</think>", "<answer>")
SIMPLIFICATION_MAX_BATCH_SIZE = 4
SIMPLIFICATION_OUTPUT_FIELDS = ("id", "problem", "reasoning", "answer")
SIMPLIFICATION_SYSTEM_PROMPT = """You are a fidelity-first curriculum compressor for reasoning-SFT examples that must fit a ~100M parameter model with a 2,048-token context window.

Process every input item independently and return one rewritten item for each input id.

Priority order:
1. Preserve the original task type, domain, conclusion, named entities, factual relationships, constraints, and numerical givens whenever they can fit.
2. First shorten by removing repetition, digressions, verbose exposition, redundant examples, and unnecessary formatting. Make the reasoning explicit, linear, and easy to imitate.
3. If the ORIGINAL REQUIRED OUTPUT is itself too large for a compact example, make the smallest scope reduction necessary: shorten source material, reduce repeated list items/subparts, or narrow the requested deliverable while preserving the same instruction-following skill and core reasoning pattern. The answer must then be correct for the rewritten problem.
4. Do NOT change scientific constants, numerical values, entities, or conclusions merely to make an example look simpler. Change them only if unavoidable for a scope-reduced analogous task, and prefer deleting irrelevant context instead.
5. Never invent unsupported facts. Keep each problem self-contained and solvable.
6. Target <= 160 words for problem, <= 300 words for reasoning, <= 160 words for answer, preferably <= 600 words total. Concision is more important than filling the budget.
7. Output strict RFC 8259 JSON only: one JSON array, same item order, objects with exactly id, problem, reasoning, answer. Use valid JSON escaping. Prefer plain-text math notation instead of LaTeX/backslashes. No Markdown code fence and no commentary outside the JSON."""

_MATH_PRIMARY_TOPIC = re.compile(
    r"\b(?:algebra|algebraic|calculus|derivative|differentiate|integral|integrate|"
    r"trigonometry|trigonometric|geometry|pythagorean|quadratic|polynomial|factorial|"
    r"combinatorics|combinatorial|matrix|matrices|determinant|eigenvalue|logarithm|"
    r"logarithmic|arithmetic|number theory|group theory|homomorphism|theorem|proof|"
    r"probability distribution|standard deviation|variance|equation|inequality|fractions?)\b",
    re.IGNORECASE,
)
_MATH_PRIMARY_ACTION = re.compile(
    r"\b(?:calculate|compute|solve|evaluate|simplify|derive|differentiate|integrate|find)\b",
    re.IGNORECASE,
)
_MATH_NOTATION = re.compile(
    r"(?:\d\s*[%+*/^=]|[%+*/^=]\s*\d|\$\s*\d|"
    r"\\(?:frac|sqrt|boxed|sum|int|pi|theta)|\b(?:sin|cos|tan)\s*\()",
    re.IGNORECASE,
)
_CODE_LANGUAGE = re.compile(
    r"\b(?:python|javascript|typescript|java\b|c\+\+|c#|golang|rust\b|sql\b|"
    r"html\b|css\b|react\b|node\.js|pandas|numpy|tensorflow|pytorch|matlab|maple|"
    r"powershell|bash\b|shell script|regex)\b",
    re.IGNORECASE,
)
_CODE_TASK = re.compile(
    r"(?:\b(?:write|create|implement|debug|fix|convert|generate|provide)\b.{0,80}"
    r"\b(?:code|script|function|algorithm|query|computer program)\b|"
    r"\b(?:code|script|function|algorithm|query|computer program)\b.{0,80}"
    r"\b(?:write|create|implement|debug|fix|convert|generate|provide)\b)",
    re.IGNORECASE | re.DOTALL,
)
_CODE_LITERAL = re.compile(
    r"```|\bdef\s+\w+\s*\(|\bclass\s+\w+\s*[:({]|"
    r"\bfrom\s+\w+(?:\.\w+)*\s+import\s+\w+|"
    r"\b(?:const|let|var)\s+\w+\s*=|\bSELECT\s+.+\s+FROM\b|"
    r"\bfunction\s+\w*\s*\(",
    re.IGNORECASE,
)


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
            "dataset_revision": DATASET_REVISION,
            "dataset_filename": DATASET_FILENAME,
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


@dataclass(frozen=True, slots=True)
class CleanInstructionReport:
    source_rows: int
    domain_counts: dict[str, int]
    valid_unique_instruction_rows: int
    rejected_output_count: int
    duplicate_input_count: int
    exclusion_counts: dict[str, int]
    selected_count: int
    selected_min_serialized_tokens: int | None
    selected_max_serialized_tokens: int | None

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset_id": DATASET_ID,
            "dataset_config": DATASET_CONFIG,
            "dataset_split": DATASET_SPLIT,
            "dataset_revision": DATASET_REVISION,
            "dataset_filename": DATASET_FILENAME,
            "source_rows": self.source_rows,
            "domain_counts": dict(sorted(self.domain_counts.items())),
            "valid_unique_instruction_rows": self.valid_unique_instruction_rows,
            "rejected_output_count": self.rejected_output_count,
            "duplicate_input_count": self.duplicate_input_count,
            "exclusion_counts": dict(sorted(self.exclusion_counts.items())),
            "selected_count": self.selected_count,
            "selected_min_serialized_tokens": self.selected_min_serialized_tokens,
            "selected_max_serialized_tokens": self.selected_max_serialized_tokens,
        }


def parse_teacher_output(output: str) -> ParsedOutput:
    """Extract a non-empty reasoning body and the final answer."""

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


def build_simplification_messages(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, str]]:
    """Build one low-volume GemRouter request for over-context examples."""

    if not records:
        raise ValueError("simplification batch cannot be empty")
    if len(records) > SIMPLIFICATION_MAX_BATCH_SIZE:
        raise ValueError(
            f"simplification batch cannot exceed {SIMPLIFICATION_MAX_BATCH_SIZE} examples"
        )

    payload: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        item_id = _row_text(record, "id", index=index)
        if item_id in seen_ids:
            raise ValueError(f"simplification batch contains duplicate id {item_id!r}")
        seen_ids.add(item_id)
        item = {
            "id": item_id,
            "problem": _row_text(record, "problem", index=index),
            "reasoning": _row_text(record, "reasoning", index=index),
            "answer": _row_text(record, "answer", index=index),
        }
        skill = record.get("skill")
        if skill is not None:
            if not isinstance(skill, str) or not skill.strip():
                raise ValueError(f"Superior row {index} field 'skill' must be non-empty text")
            item["skill"] = skill.strip()
        payload.append(item)

    return (
        {"role": "system", "content": SIMPLIFICATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        },
    )


def parse_simplification_response(
    text: str,
    *,
    expected_ids: Sequence[str],
) -> tuple[dict[str, str], ...]:
    """Strictly parse one batched simplification response in input order."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("simplification response must be non-empty text")
    if not expected_ids:
        raise ValueError("expected_ids cannot be empty")
    if len(expected_ids) > SIMPLIFICATION_MAX_BATCH_SIZE:
        raise ValueError(
            f"expected_ids cannot exceed {SIMPLIFICATION_MAX_BATCH_SIZE} examples"
        )
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("expected_ids must be unique")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("simplification response must be strict JSON with no surrounding text") from error
    if not isinstance(raw, list):
        raise ValueError("simplification response must be a JSON array")
    if len(raw) != len(expected_ids):
        raise ValueError(
            f"simplification response returned {len(raw)} records; expected {len(expected_ids)}"
        )

    result: list[dict[str, str]] = []
    expected_fields = set(SIMPLIFICATION_OUTPUT_FIELDS)
    for index, (item, expected_id) in enumerate(zip(raw, expected_ids)):
        if not isinstance(item, Mapping) or set(item) != expected_fields:
            raise ValueError(
                f"simplification item {index} must contain exactly {sorted(expected_fields)}"
            )
        normalized: dict[str, str] = {}
        for field in SIMPLIFICATION_OUTPUT_FIELDS:
            value = item[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"simplification item {index} field {field!r} must be non-empty text"
                )
            normalized[field] = value.strip()
        if normalized["id"] != expected_id:
            raise ValueError(
                f"simplification item {index} id {normalized['id']!r} does not match {expected_id!r}"
            )
        result.append(normalized)
    return tuple(result)


def _default_token_counter() -> Callable[[str], int]:
    try:
        import tiktoken
    except ImportError as error:  # pragma: no cover - environment dependency
        raise RuntimeError("tiktoken is required for GPT-2 reasoning-length ranking") from error
    encoding = tiktoken.get_encoding("gpt2")

    def count(text: str) -> int:
        return len(encoding.encode(text, allowed_special=set(), disallowed_special=()))

    return count


def instruction_exclusion_reason(problem: str) -> str | None:
    """Return a conservative primary-task exclusion for production R0.

    Formatting constraints such as "at least 5 bullets" or incidental percentages
    are intentionally allowed. We exclude explicit mathematical topics, explicit
    compute/solve requests paired with mathematical notation, and programming
    tasks/languages. When both categories match, the overlap is recorded.
    """

    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("problem must be non-empty text")
    math_primary = bool(
        _MATH_PRIMARY_TOPIC.search(problem)
        or (_MATH_PRIMARY_ACTION.search(problem) and _MATH_NOTATION.search(problem))
    )
    code_primary = bool(
        _CODE_LANGUAGE.search(problem)
        or _CODE_TASK.search(problem)
        or _CODE_LITERAL.search(problem)
    )
    if math_primary and code_primary:
        return "math_and_code_primary"
    if math_primary:
        return "math_primary"
    if code_primary:
        return "code_primary"
    return None


def atomic_rsft_serialized_tokens(
    *,
    problem: str,
    reasoning: str,
    answer: str,
    token_counter: Callable[[str], int] | None = None,
) -> int:
    """Count the exact loss-bearing R-SFT chat sequence under atomic markers.

    ``GPT2ChatTemplate`` excludes its leading BOS/EOS sentinel from the model
    context. The remaining sequence is ``User:\n`` + problem + ``\n\nAssistant:\n``
    + three promoted one-token control markers + reasoning + answer + final EOS.
    """

    for name, value in (("problem", problem), ("reasoning", reasoning), ("answer", answer)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be non-empty text")
    count = token_counter or _default_token_counter()
    return (
        count("User:\n")
        + count(problem)
        + count("\n\nAssistant:\n")
        + 3
        + count(reasoning)
        + count(answer)
        + 1
    )


def select_clean_instruction(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed: int = DEFAULT_SEED,
    context_length: int = PRODUCTION_CONTEXT_LENGTH,
    token_counter: Callable[[str], int] | None = None,
) -> tuple[list[dict[str, Any]], CleanInstructionReport]:
    """Keep every unique, non-math/non-code instruction row that fits R-SFT."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(context_length, bool) or not isinstance(context_length, int) or context_length <= 0:
        raise ValueError("context_length must be a positive integer")
    count = token_counter or _default_token_counter()

    source_rows = 0
    domain_counts: Counter[str] = Counter()
    exclusion_counts: Counter[str] = Counter()
    rejected_output_count = 0
    duplicate_input_count = 0
    valid_unique_instruction_rows = 0
    seen_inputs: set[str] = set()
    selected: list[dict[str, Any]] = []

    for source_index, row in enumerate(rows):
        source_rows += 1
        if not isinstance(row, Mapping):
            raise ValueError(f"Superior row {source_index} must be an object")
        raw_domain = row.get("domain")
        domain = raw_domain.strip() if isinstance(raw_domain, str) else "<invalid>"
        domain_counts[domain] += 1
        if domain != PRODUCTION_DOMAIN:
            continue

        problem = _row_text(row, "input", index=source_index)
        source_id = _row_text(row, "uuid", index=source_index)
        output = _row_text(row, "output", index=source_index)
        try:
            parsed = parse_teacher_output(output)
        except ValueError:
            rejected_output_count += 1
            continue

        prompt_hash = _normalized_input_hash(problem)
        if prompt_hash in seen_inputs:
            duplicate_input_count += 1
            continue
        seen_inputs.add(prompt_hash)
        valid_unique_instruction_rows += 1

        exclusion = instruction_exclusion_reason(problem)
        if exclusion is not None:
            exclusion_counts[exclusion] += 1
            continue
        if any(
            marker in text
            for marker in PRODUCTION_RESERVED_MARKERS
            for text in (problem, parsed.reasoning, parsed.answer)
        ):
            exclusion_counts["reserved_marker_collision"] += 1
            continue

        serialized_tokens = atomic_rsft_serialized_tokens(
            problem=problem,
            reasoning=parsed.reasoning,
            answer=parsed.answer,
            token_counter=count,
        )
        if serialized_tokens > context_length:
            exclusion_counts["over_context"] += 1
            continue

        selected.append(
            {
                "source_index": source_index,
                "source_id": source_id,
                "domain": domain,
                "difficulty": PRODUCTION_DIFFICULTY,
                "problem": problem,
                "reasoning": parsed.reasoning,
                "answer": parsed.answer,
                "serialized_token_count": serialized_tokens,
            }
        )

    selected.sort(
        key=lambda item: hashlib.sha256(
            f"{seed}\x1f{item['source_id']}".encode("utf-8")
        ).hexdigest()
    )
    token_counts = [int(item["serialized_token_count"]) for item in selected]
    report = CleanInstructionReport(
        source_rows=source_rows,
        domain_counts=dict(domain_counts),
        valid_unique_instruction_rows=valid_unique_instruction_rows,
        rejected_output_count=rejected_output_count,
        duplicate_input_count=duplicate_input_count,
        exclusion_counts=dict(exclusion_counts),
        selected_count=len(selected),
        selected_min_serialized_tokens=min(token_counts) if token_counts else None,
        selected_max_serialized_tokens=max(token_counts) if token_counts else None,
    )
    return selected, report


def iter_stage1_rows(
    *, timeout_seconds: float = DEFAULT_HTTP_TIMEOUT_SECONDS
) -> Iterator[Mapping[str, Any]]:
    """Stream the public 4.6 GB Stage-1 JSONL without downloading it first."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    request = Request(
        DATASET_URL,
        headers={"User-Agent": "small-llm-superior-reasoning/1"},
        method="GET",
    )
    try:
        response = urlopen(request, timeout=timeout_seconds)
    except OSError as error:  # pragma: no cover - network path
        raise RuntimeError("failed to open Superior Reasoning Stage-1 stream") from error

    with response, io.TextIOWrapper(response, encoding="utf-8") as text_stream:
        for line_number, line in enumerate(text_stream, start=1):
            if not line.strip():
                raise ValueError(f"Superior source contains a blank line at {line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Superior source line {line_number} is invalid JSON") from error
            if not isinstance(row, Mapping):
                raise ValueError(f"Superior source line {line_number} must be a JSON object")
            yield row


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
    handle.write(
        json.dumps(
            {
                "source_index": source_index,
                "source_id": source_id,
                "domain": domain,
                "problem": problem,
                "reasoning": reasoning,
                "answer": answer,
                "reasoning_token_count": reasoning_token_count,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    )
    return offset


def _push_shortest(
    heap: list[tuple[int, int, CandidateRef]],
    candidate: CandidateRef,
    *,
    capacity: int,
) -> None:
    entry = (-candidate.reasoning_token_count, -candidate.source_index, candidate)
    if len(heap) < capacity:
        heapq.heappush(heap, entry)
        return
    worst = (-heap[0][0], -heap[0][1])
    current = (candidate.reasoning_token_count, candidate.source_index)
    if current < worst:
        heapq.heapreplace(heap, entry)


def _allocate_counts(
    *, total: int, science_share: float, usable_counts: Mapping[str, int]
) -> dict[str, int]:
    if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
        raise ValueError("total must be a positive integer")
    if not 0.0 <= science_share <= 1.0:
        raise ValueError("science_share must be in [0, 1]")

    science_target = round(total * science_share)
    requested = {
        "science": science_target,
        "instruction_following": total - science_target,
    }
    counts = {
        domain: min(requested[domain], int(usable_counts.get(domain, 0)))
        for domain in ELIGIBLE_DOMAINS
    }
    deficit = total - sum(counts.values())
    domains = sorted(
        ELIGIBLE_DOMAINS,
        key=lambda domain: (
            int(usable_counts.get(domain, 0)) - counts[domain],
            domain == "instruction_following",
        ),
        reverse=True,
    )
    for domain in domains:
        if deficit == 0:
            break
        available = max(0, int(usable_counts.get(domain, 0)) - counts[domain])
        take = min(deficit, available)
        counts[domain] += take
        deficit -= take
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
    return hashlib.sha256(f"{seed}\x1f{candidate.source_id}".encode("utf-8")).hexdigest()


def select_shortest(
    rows: Iterable[Mapping[str, Any]],
    *,
    total: int = DEFAULT_TOTAL,
    science_share: float = DEFAULT_SCIENCE_SHARE,
    seed: int = DEFAULT_SEED,
    token_counter: Callable[[str], int] | None = None,
    spool_dir: str | Path | None = None,
) -> tuple[list[dict[str, Any]], SelectionReport]:
    """Count the full stream and retain the shortest eligible unique traces."""

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
            raw_domain = row.get("domain")
            domain = raw_domain.strip() if isinstance(raw_domain, str) else "<invalid>"
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

            prompt_hash = _normalized_input_hash(problem)
            if prompt_hash in seen_inputs:
                duplicate_input_count += 1
                continue
            seen_inputs.add(prompt_hash)

            token_count = int(token_counter(parsed.reasoning))
            if token_count <= 0:
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
                reasoning_token_count=token_count,
            )
            _push_shortest(
                heaps[domain],
                CandidateRef(offset, source_index, source_id, domain, token_count),
                capacity=total,
            )

        selected_counts = _allocate_counts(
            total=total,
            science_share=science_share,
            usable_counts=usable_counts,
        )
        selected_refs: list[CandidateRef] = []
        for domain in ELIGIBLE_DOMAINS:
            ranked = sorted(
                (entry[2] for entry in heaps[domain]),
                key=lambda item: (item.reasoning_token_count, item.source_index),
            )
            selected_refs.extend(ranked[: selected_counts[domain]])
        selected_refs.sort(key=lambda item: _selection_order_key(item, seed=seed))
        selected = [_read_spooled(spool, item.offset) for item in selected_refs]

    token_counts = [item.reasoning_token_count for item in selected_refs]
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
        realized_science_share=(selected_counts["science"] / len(selected)) if selected else 0.0,
        selected_min_reasoning_tokens=min(token_counts) if token_counts else None,
        selected_max_reasoning_tokens=max(token_counts) if token_counts else None,
    )
    return selected, report


def to_rsft_mapping(record: Mapping[str, Any]) -> dict[str, str]:
    domain = str(record["domain"])
    if domain not in SOURCE_SKILLS:
        raise ValueError(f"unsupported Superior domain {domain!r}")
    difficulty = record.get("difficulty", SOURCE_DIFFICULTY)
    if not isinstance(difficulty, str) or not difficulty.strip():
        raise ValueError("Superior difficulty label must be non-empty text")
    return {
        "skill": SOURCE_SKILLS[domain],
        "difficulty": difficulty.strip(),
        "problem": str(record["problem"]).strip(),
        "reasoning": str(record["reasoning"]).strip(),
        "answer": str(record["answer"]).strip(),
    }


def _read_gemini_jsonl(path: str | Path) -> list[dict[str, str]]:
    expected = {"skill", "difficulty", "problem", "reasoning", "answer"}
    records: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Gemini JSONL contains a blank line at {line_number}")
            raw = json.loads(line)
            if not isinstance(raw, Mapping) or set(raw) != expected:
                raise ValueError(
                    f"Gemini JSONL line {line_number} must contain exactly {sorted(expected)}"
                )
            record: dict[str, str] = {}
            for field in expected:
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
    combined = [to_rsft_mapping(record) for record in superior_records]
    if gemini_jsonl is not None:
        combined.extend(_read_gemini_jsonl(gemini_jsonl))
    random.Random(seed).shuffle(combined)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in combined:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    return destination


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_production_dataset(
    output_jsonl: str | Path,
    *,
    gemini_jsonl: str | Path | None = None,
    seed: int = DEFAULT_SEED,
    context_length: int = PRODUCTION_CONTEXT_LENGTH,
) -> dict[str, object]:
    """Build the accepted instruction-only, context-safe R0 reasoning corpus."""

    selected, report = select_clean_instruction(
        iter_stage1_rows(),
        seed=seed,
        context_length=context_length,
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
            "schema": "small-llm-superior-reasoning-production-v1",
            "policy": PRODUCTION_FILTER_VERSION,
            "production_domain": PRODUCTION_DOMAIN,
            "context_length": context_length,
            "gemini_rows": gemini_rows,
            "combined_rows": len(selected) + gemini_rows,
            "output_jsonl": str(destination),
            "output_sha256": _sha256(destination),
            "seed": seed,
            "selection_policy": "all-valid-unique-clean-context-fit-instruction-rows",
            "math_policy": "exclude-primary-computation-but-allow-incidental-numbers-and-format-counts",
            "code_policy": "exclude-primary-programming-and-code-generation-tasks",
        }
    )
    manifest = destination.with_suffix(destination.suffix + ".manifest.json")
    manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["manifest"] = str(manifest)
    return result


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
    counts: Counter[str] = Counter()
    rows = 0
    for index, row in enumerate(iter_stage1_rows()):
        rows += 1
        if not isinstance(row, Mapping):
            raise ValueError(f"Superior row {index} must be an object")
        raw_domain = row.get("domain")
        domain = raw_domain.strip() if isinstance(raw_domain, str) else "<invalid>"
        counts[domain] += 1
    return {
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_split": DATASET_SPLIT,
        "dataset_revision": DATASET_REVISION,
        "dataset_filename": DATASET_FILENAME,
        "rows": rows,
        "domain_counts": dict(sorted(counts.items())),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("scan", help="print exact Stage-1 domain counts")
    production = subparsers.add_parser(
        "build-production",
        help="build the accepted instruction-only, non-math/non-code, context-fit corpus",
    )
    production.add_argument("--output-jsonl", type=Path, required=True)
    production.add_argument("--gemini-jsonl", type=Path)
    production.add_argument("--seed", type=int, default=DEFAULT_SEED)
    production.add_argument("--context-length", type=int, default=PRODUCTION_CONTEXT_LENGTH)

    build = subparsers.add_parser(
        "build-legacy-mixed",
        help="reproduce the superseded ADR 0102 30/70 science/instruction selection",
    )
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
    if args.command == "build-production":
        result = build_production_dataset(
            args.output_jsonl,
            gemini_jsonl=args.gemini_jsonl,
            seed=args.seed,
            context_length=args.context_length,
        )
    else:
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
