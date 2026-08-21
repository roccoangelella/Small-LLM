"""Unified Superior Reasoning source adapter for R-SFT dataset production.

This is the active Superior source implementation. Stage 1 and Stage 2 share one
normalization/filtering contract and differ only by their upstream file identity.
Historical one-off Stage-1/Stage-2 scripts remain in the repository for artifact
reproduction but are not the active multi-source build path.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import importlib.util
import io
import json
from pathlib import Path
import re
import sys
from types import ModuleType
from typing import Any, Callable, Iterator, Mapping, Sequence
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
DATASET_DIR = HERE.parent


def _load_common() -> ModuleType:
    name = "small_llm_rsft_dataset_common"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    path = DATASET_DIR / "common.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load R-SFT common module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


common = _load_common()

SOURCE_NAME = "superior_reasoning"
DATASET_ID = "Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b"
DATASET_REVISION = "main"
DEFAULT_HTTP_TIMEOUT_SECONDS = 120.0
PRODUCTION_DOMAIN = "instruction_following"
PRODUCTION_SKILL = "SR_INSTRUCTION_FOLLOWING"
FIT_DIFFICULTY = "clean_fit"
ADAPTED_DIFFICULTY = "simplified_fit"
FILTER_VERSION = "instruction-no-math-code-v1"


@dataclass(frozen=True, slots=True)
class StageSpec:
    name: str
    filename: str
    priority: int

    @property
    def url(self) -> str:
        return (
            f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/"
            f"{self.filename}?download=true"
        )


STAGES: dict[str, StageSpec] = {
    "stage1": StageSpec(
        "stage1",
        "Superior-Reasoning-SFT-gpt-oss-120b-stage1-train-data.jsonl",
        1,
    ),
    "stage2": StageSpec(
        "stage2",
        "Superior-Reasoning-SFT-gpt-oss-120b-stage2-train-data.jsonl",
        2,
    ),
}
DEFAULT_STAGES = ("stage2",)

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


def parse_teacher_output(output: str) -> ParsedOutput:
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


def instruction_exclusion_reason(problem: str) -> str | None:
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


def _row_text(row: Mapping[str, Any], field: str, *, index: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Superior row {index} field {field!r} must be non-empty text")
    return value.strip()


def _iter_local_jsonl(path: Path) -> Iterator[Mapping[str, Any]]:
    for row in common.read_jsonl(path):
        yield row


def iter_stage_rows(
    stage: str,
    *,
    source_jsonl: Path | None = None,
    timeout_seconds: float = DEFAULT_HTTP_TIMEOUT_SECONDS,
) -> Iterator[Mapping[str, Any]]:
    """Stream one Superior stage from a local JSONL or its canonical HF file."""
    try:
        spec = STAGES[stage]
    except KeyError as error:
        raise ValueError(f"unsupported Superior stage {stage!r}; choose {sorted(STAGES)}") from error
    if source_jsonl is not None:
        yield from _iter_local_jsonl(Path(source_jsonl))
        return
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    request = Request(
        spec.url,
        headers={"User-Agent": "small-llm-rsft-dataset/2"},
        method="GET",
    )
    try:
        response = urlopen(request, timeout=timeout_seconds)
    except OSError as error:  # pragma: no cover - network path
        raise RuntimeError(f"failed to open Superior Reasoning {stage} stream") from error
    with response, io.TextIOWrapper(response, encoding="utf-8") as text_stream:
        for line_number, line in enumerate(text_stream, start=1):
            if not line.strip():
                raise RuntimeError(f"Superior {stage} contains a blank line at {line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Superior {stage} line {line_number} is invalid JSON") from error
            if not isinstance(row, Mapping):
                raise RuntimeError(f"Superior {stage} line {line_number} must be an object")
            yield row


def prepare(
    *,
    output_dir: Path,
    stages: Sequence[str] = DEFAULT_STAGES,
    base_prompt_hashes: set[str] | None = None,
    source_jsonl_by_stage: Mapping[str, Path] | None = None,
    context_length: int = common.CONTEXT_LENGTH,
    token_counter: Callable[[str], int] | None = None,
    progress_every: int = 5_000,
) -> dict[str, object]:
    """Normalize/filter Superior stages into generic fit and over-context streams."""
    if not stages:
        raise ValueError("at least one Superior stage is required")
    normalized_stages = tuple(dict.fromkeys(str(stage).strip() for stage in stages))
    for stage in normalized_stages:
        if stage not in STAGES:
            raise ValueError(f"unsupported Superior stage {stage!r}")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    fit_path = destination / "fit.jsonl"
    candidates_path = destination / "candidates.jsonl"
    manifest_path = destination / "manifest.json"
    if any(path.exists() for path in (fit_path, candidates_path, manifest_path)):
        raise FileExistsError(f"refusing to replace prepared source directory {destination}")

    count_tokens = token_counter or common.default_token_counter()
    seen = set(base_prompt_hashes or ())
    source_paths = {str(k): Path(v) for k, v in (source_jsonl_by_stage or {}).items()}
    fit_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    stage_reports: dict[str, object] = {}

    for stage in normalized_stages:
        domain_counts: Counter[str] = Counter()
        exclusion_counts: Counter[str] = Counter()
        source_rows = instruction_rows = rejected_output = duplicates = 0
        fit_count = candidate_count = 0
        spec = STAGES[stage]
        for source_index, raw in enumerate(
            iter_stage_rows(stage, source_jsonl=source_paths.get(stage))
        ):
            source_rows += 1
            if progress_every and source_rows % progress_every == 0:
                print(
                    f"[superior:{stage}] source={source_rows} fit={fit_count} "
                    f"over_context={candidate_count} duplicates={duplicates}",
                    flush=True,
                )
            raw_domain = raw.get("domain")
            domain = raw_domain.strip() if isinstance(raw_domain, str) else "<invalid>"
            domain_counts[domain] += 1
            if domain != PRODUCTION_DOMAIN:
                continue
            instruction_rows += 1

            problem = _row_text(raw, "input", index=source_index)
            source_id = _row_text(raw, "uuid", index=source_index)
            output = _row_text(raw, "output", index=source_index)
            try:
                parsed = parse_teacher_output(output)
            except ValueError:
                rejected_output += 1
                continue

            prompt_hash = common.normalized_prompt_hash(problem)
            if prompt_hash in seen:
                duplicates += 1
                continue
            seen.add(prompt_hash)

            exclusion = instruction_exclusion_reason(problem)
            if exclusion is not None:
                exclusion_counts[exclusion] += 1
                continue
            try:
                common.validate_reasoning_text(
                    {"problem": problem, "reasoning": parsed.reasoning, "answer": parsed.answer}
                )
            except ValueError:
                exclusion_counts["reserved_marker_collision"] += 1
                continue

            serialized = common.atomic_rsft_serialized_tokens(
                problem=problem,
                reasoning=parsed.reasoning,
                answer=parsed.answer,
                token_counter=count_tokens,
            )
            source_order = spec.priority * 1_000_000_000 + source_index
            base = {
                "id": source_id,
                "source": SOURCE_NAME,
                "source_stage": stage,
                "source_index": source_index,
                "source_order": source_order,
                "skill": PRODUCTION_SKILL,
                "difficulty": FIT_DIFFICULTY,
                "problem": problem,
                "reasoning": parsed.reasoning,
                "answer": parsed.answer,
            }
            if serialized <= context_length:
                fit_rows.append(
                    {
                        **base,
                        "schema": common.FIT_SCHEMA,
                        "serialized_token_count": serialized,
                        "target_token_count": common.assistant_target_tokens(
                            reasoning=parsed.reasoning,
                            answer=parsed.answer,
                            token_counter=count_tokens,
                        ),
                    }
                )
                fit_count += 1
            else:
                candidate_rows.append(
                    {
                        **base,
                        "schema": common.OVER_CONTEXT_SCHEMA,
                        "difficulty": ADAPTED_DIFFICULTY,
                        "original_serialized_tokens": serialized,
                    }
                )
                exclusion_counts["over_context"] += 1
                candidate_count += 1

        stage_reports[stage] = {
            "filename": spec.filename,
            "source_rows": source_rows,
            "instruction_rows": instruction_rows,
            "fit_rows": fit_count,
            "over_context_rows": candidate_count,
            "rejected_output_rows": rejected_output,
            "duplicate_or_base_collision_rows": duplicates,
            "domain_counts": dict(sorted(domain_counts.items())),
            "exclusion_counts": dict(sorted(exclusion_counts.items())),
        }

    fit_rows.sort(key=lambda row: (int(row["source_order"]), str(row["id"])))
    candidate_rows.sort(key=lambda row: (int(row["source_order"]), str(row["id"])))
    common.write_jsonl(fit_path, fit_rows)
    common.write_jsonl(candidates_path, candidate_rows)
    manifest = {
        "schema": common.SOURCE_MANIFEST_SCHEMA,
        "source": SOURCE_NAME,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "filter_version": FILTER_VERSION,
        "context_length": context_length,
        "stages": list(normalized_stages),
        "stage_reports": stage_reports,
        "fit": {
            "path": fit_path.name,
            "records": len(fit_rows),
            "sha256": common.sha256_path(fit_path),
            "byte_size": fit_path.stat().st_size,
        },
        "over_context": {
            "path": candidates_path.name,
            "records": len(candidate_rows),
            "sha256": common.sha256_path(candidates_path),
            "byte_size": candidates_path.stat().st_size,
        },
    }
    common.atomic_json(manifest_path, manifest)
    return {**manifest, "manifest": str(manifest_path)}
