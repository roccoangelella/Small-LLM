from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "post_training" / "R-SFT" / "dataset" / "scale_superior_reasoning.py"


def _load_module():
    name = "test_small_llm_stage2_scaling"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def _base_row(i: int, *, reasoning_words: int = 4) -> dict[str, str]:
    return {
        "skill": "SR_INSTRUCTION_FOLLOWING",
        "difficulty": "clean_fit",
        "problem": f"Base prompt {i}",
        "reasoning": " ".join(["reason"] * reasoning_words),
        "answer": f"answer {i}",
    }


def _source_row(uuid: str, problem: str, reasoning: str, answer: str = "ok") -> dict[str, str]:
    return {
        "uuid": uuid,
        "domain": "instruction_following",
        "input": problem,
        "output": f"<think>{reasoning}</think>{answer}",
    }


def test_prepare_stage2_reuses_stage1_filter_and_context_contract(tmp_path, monkeypatch):
    module = _load_module()
    base = tmp_path / "base.jsonl"
    _write_jsonl(base, [_base_row(i) for i in range(6)])

    rows = [
        _source_row("fit", "Summarize this short note.", "short reasoning"),
        _source_row("math", "Solve the algebra equation x = 2 + 2.", "math reasoning"),
        _source_row("code", "Write Python code that prints hello.", "code reasoning"),
        _source_row("duplicate", "Base prompt 0", "duplicate reasoning"),
        _source_row("long", "Rewrite this document briefly.", "x " * 2_100),
    ]
    monkeypatch.setattr(module, "iter_stage2_instruction_rows", lambda source_jsonl=None: iter(rows))
    monkeypatch.setattr(
        module.superior,
        "_default_token_counter",
        lambda: (lambda text: len(text.split())),
    )

    result = module.prepare_stage2(base_jsonl=base, work_dir=tmp_path / "work")
    assert result["fit_unchanged_rows"] == 1
    assert result["over_context_rows"] == 1
    assert result["duplicate_or_base_collision_count"] == 1
    assert result["exclusion_counts"]["math_primary"] == 1
    assert result["exclusion_counts"]["code_primary"] == 1

    fit = list(module._read_jsonl(tmp_path / "work" / "fit.jsonl"))
    candidates = list(module._read_jsonl(tmp_path / "work" / "candidates.jsonl"))
    assert [row["id"] for row in fit] == ["fit"]
    assert [row["id"] for row in candidates] == ["long"]
    assert fit[0]["serialized_token_count"] <= 2_048
    assert candidates[0]["original_serialized_tokens"] > 2_048


def test_build_percent_uses_projected_train_reasoning_targets(tmp_path, monkeypatch):
    module = _load_module()
    monkeypatch.setattr(
        module.superior,
        "_default_token_counter",
        lambda: (lambda text: len(text.split())),
    )
    # Small synthetic parent: 1% = 100 total targets, 90 reasoning / 10 retention.
    monkeypatch.setattr(module, "PARENT_TRAIN_TARGETS", 10_000)

    base = tmp_path / "base.jsonl"
    _write_jsonl(base, [_base_row(i, reasoning_words=2) for i in range(10)])

    work = tmp_path / "work"
    work.mkdir()
    fit_rows = []
    for i in range(30):
        fit_rows.append(
            {
                "id": f"s2-{i}",
                "source_index": i,
                "domain": "instruction_following",
                "difficulty": "clean_fit",
                "problem": f"Stage two prompt {i}",
                "reasoning": "reason reason",
                "answer": f"answer {i}",
                "serialized_token_count": 20,
                "target_token_count": 8,
            }
        )
    _write_jsonl(work / "fit.jsonl", fit_rows)

    output = tmp_path / "one-percent.jsonl"
    result = module.build_percent_corpus(
        base_jsonl=base,
        work_dir=work,
        output_jsonl=output,
        percent=1.0,
    )
    assert result["requested_total_train_targets"] == 100
    assert result["requested_reasoning_train_targets"] == 90
    assert result["projected_reasoning_train_targets"] >= 90
    assert result["stage2_rows_added"] > 0
    assert result["combined_rows"] == 10 + result["stage2_rows_added"]
    assert output.is_file()
    assert output.with_suffix(".jsonl.manifest.json").is_file()
