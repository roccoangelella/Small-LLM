from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "post_training" / "R-SFT" / "build_atomic.py"


def _load_module():
    name = "test_small_llm_rsft_build_atomic"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class _Record:
    skill: str
    difficulty: str


def _uniform_records(module, count: int) -> list[_Record]:
    return [
        _Record(skill=skill, difficulty=difficulty)
        for skill in module.bundle.prompts.R0_SKILLS
        for difficulty in module.bundle.generation.R0_DIFFICULTIES
        for _ in range(count)
    ]


def test_canonical_production_token_spec_is_atomic() -> None:
    module = _load_module()
    spec = module._canonical_token_spec()
    assert (spec.reasoning_start, spec.reasoning_end, spec.answer_start) == (
        "<think>",
        "</think>",
        "<answer>",
    )
    assert spec.special_tokens == {
        "<think>": 50_257,
        "</think>": 50_258,
        "<answer>": 50_259,
    }


def test_production_matrix_size_is_inferred_but_must_be_uniform() -> None:
    module = _load_module()
    records = _uniform_records(module, 37)
    assert module._infer_examples_per_cell(records) == 37

    records.pop()
    with pytest.raises(ValueError):
        module._infer_examples_per_cell(records)


def test_atomic_retention_target_uses_90_10_top_level_mix() -> None:
    module = _load_module()
    reasoning = 900_000
    assert module._retention_target(reasoning) == 100_000


def test_production_builder_requires_explicit_heldout_scale() -> None:
    module = _load_module()
    parser = module.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--reasoning-jsonl",
                "reasoning.jsonl",
                "--s0-bundle",
                "s0",
                "--output-dir",
                "out",
            ]
        )
