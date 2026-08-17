"""Generate the first R0 reasoning-SFT corpus through the Gemini teacher.

The generator deliberately trusts teacher semantics. Its rejection boundary is
strict JSON/schema conformance only. Difficulty labels remain project-side
metadata: Gemini receives the corresponding structural description, never L1/L2/L3.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import random
import sys
from types import ModuleType
from typing import Any, Protocol, Sequence


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


dataset = _load_sibling("dataset")
prompts = _load_sibling("prompts")
schema = _load_sibling("schema")

R0_DIFFICULTIES = ("L1", "L2", "L3")
DEFAULT_SEED = 17

# These descriptions are teacher-facing. The internal level names are never
# interpolated into the prompt itself.
DIFFICULTY_REQUIREMENTS = {
    "L1": (
        "Keep each problem structurally simple and local. The requested conclusion should follow "
        "from one compact reasoning interaction without a long dependency chain or search."
    ),
    "L2": (
        "Require a moderate composition of dependent information. The answer should require "
        "combining multiple relevant facts, relations, observations, or constraints rather than "
        "reading it from one isolated statement, while remaining compact."
    ),
    "L3": (
        "Require deeper composition across several interacting pieces of information. Small "
        "branching, elimination, or reuse of intermediate conclusions is appropriate when natural, "
        "but keep the problem clear and self-contained rather than turning it into a large search."
    ),
}


class TeacherClient(Protocol):
    def complete_text(self, prompt: str) -> Any: ...


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    skill: str
    difficulty: str
    count: int
    prompt: str


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def build_uniform_generation_plan(
    *,
    examples_per_cell: int,
    batch_size: int = prompts.DEFAULT_BATCH_SIZE,
) -> tuple[GenerationRequest, ...]:
    """Plan exactly the same number of examples for every skill x difficulty cell."""

    _positive_int("examples_per_cell", examples_per_cell)
    _positive_int("batch_size", batch_size)
    requests: list[GenerationRequest] = []
    for skill in prompts.R0_SKILLS:
        for difficulty in R0_DIFFICULTIES:
            remaining = examples_per_cell
            while remaining > 0:
                count = min(batch_size, remaining)
                prompt = prompts.build_generation_prompt(
                    skill,
                    batch_size=count,
                    structural_requirements=DIFFICULTY_REQUIREMENTS[difficulty],
                )
                requests.append(
                    GenerationRequest(
                        skill=skill,
                        difficulty=difficulty,
                        count=count,
                        prompt=prompt,
                    )
                )
                remaining -= count
    return tuple(requests)


def _response_content(response: Any) -> str:
    content = getattr(response, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("teacher client returned no textual assistant content")
    return content


def generate_uniform_dataset(
    client: TeacherClient,
    *,
    examples_per_cell: int,
    batch_size: int = prompts.DEFAULT_BATCH_SIZE,
    seed: int = DEFAULT_SEED,
) -> tuple[Any, ...]:
    """Generate, schema-check, tag, and globally shuffle one balanced R0 corpus."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    plan = build_uniform_generation_plan(
        examples_per_cell=examples_per_cell,
        batch_size=batch_size,
    )
    records: list[Any] = []
    for request in plan:
        response = client.complete_text(request.prompt)
        teacher_records = schema.parse_teacher_batch(
            _response_content(response),
            expected_count=request.count,
        )
        records.extend(
            schema.ReasoningExample.from_teacher(
                teacher_record,
                skill=request.skill,
                difficulty=request.difficulty,
            )
            for teacher_record in teacher_records
        )

    expected_total = len(prompts.R0_SKILLS) * len(R0_DIFFICULTIES) * examples_per_cell
    if len(records) != expected_total:
        raise RuntimeError(
            f"uniform generation produced {len(records)} records; expected {expected_total}"
        )
    random.Random(seed).shuffle(records)
    return tuple(records)


def plan_summary(plan: Sequence[GenerationRequest]) -> dict[str, object]:
    cells: dict[str, int] = {}
    calls: dict[str, int] = {}
    for request in plan:
        key = f"{request.skill}/{request.difficulty}"
        cells[key] = cells.get(key, 0) + request.count
        calls[key] = calls.get(key, 0) + 1
    return {
        "cells": cells,
        "calls": calls,
        "total_examples": sum(cells.values()),
        "total_calls": len(plan),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a uniform R0 reasoning-SFT dataset")
    parser.add_argument("--examples-per-cell", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=prompts.DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-prompts", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = build_uniform_generation_plan(
        examples_per_cell=args.examples_per_cell,
        batch_size=args.batch_size,
    )
    if args.dry_run:
        print(json.dumps(plan_summary(plan), indent=2, sort_keys=True))
        if args.print_prompts:
            for index, request in enumerate(plan, start=1):
                print(f"\n--- request {index}: {request.skill}/{request.difficulty} x{request.count} ---")
                print(request.prompt)
        return 0

    if args.output is None:
        raise SystemExit("--output is required unless --dry-run is used")
    if args.output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite existing output: {args.output}; pass --force")

    client = dataset.GeminiDistillationClient()
    records = generate_uniform_dataset(
        client,
        examples_per_cell=args.examples_per_cell,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    schema.write_jsonl(records, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "records": len(records),
                "skills": list(prompts.R0_SKILLS),
                "difficulties": list(R0_DIFFICULTIES),
                "examples_per_cell": args.examples_per_cell,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_SEED",
    "DIFFICULTY_REQUIREMENTS",
    "GenerationRequest",
    "R0_DIFFICULTIES",
    "build_uniform_generation_plan",
    "generate_uniform_dataset",
    "main",
    "plan_summary",
]
