from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO / "post_training" / "R-SFT" / "dataset"


def _load(name: str, path: Path):
    module_name = f"test_rsft_refactor_{name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def _counter(text: str) -> int:
    return len(text.split())


class RsftDatasetRefactorTests(unittest.TestCase):
    def test_generic_prompt_is_byte_identical_to_frozen_variant_d(self):
        module = _load("over_context", DATASET_DIR / "over_context.py")
        self.assertEqual(
            hashlib.sha256(module.SIMPLIFICATION_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
            "4c971237585acac842ed9b5417eb4d231338a0fa813ef08a377e002a99e080b9",
        )

    def test_superior_adapter_handles_stage1_and_stage2_in_one_module(self):
        module = _load("superior", DATASET_DIR / "sources" / "superior_reasoning.py")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage1 = root / "stage1.jsonl"
            stage2 = root / "stage2.jsonl"
            _write_jsonl(
                stage1,
                [
                    {
                        "uuid": "s1-fit",
                        "domain": "instruction_following",
                        "input": "Summarize the note.",
                        "output": "<think>short reason</think>summary",
                    },
                    {
                        "uuid": "s1-code",
                        "domain": "instruction_following",
                        "input": "Write Python code that prints hello.",
                        "output": "<think>reason</think>answer",
                    },
                ],
            )
            _write_jsonl(
                stage2,
                [
                    {
                        "uuid": "s2-fit",
                        "domain": "instruction_following",
                        "input": "Rewrite this title clearly.",
                        "output": "<think>short reason</think>title",
                    },
                    {
                        "uuid": "s2-long",
                        "domain": "instruction_following",
                        "input": "Condense the document.",
                        "output": "<think>" + "reason " * 2100 + "</think>answer",
                    },
                    {
                        "uuid": "s2-math-domain",
                        "domain": "math",
                        "input": "Compute 2 + 2.",
                        "output": "<think>reason</think>4",
                    },
                ],
            )
            result = module.prepare(
                output_dir=root / "prepared",
                stages=("stage1", "stage2"),
                source_jsonl_by_stage={"stage1": stage1, "stage2": stage2},
                token_counter=_counter,
                progress_every=0,
            )
            self.assertEqual(result["fit"]["records"], 2)
            self.assertEqual(result["over_context"]["records"], 1)
            self.assertEqual(
                result["stage_reports"]["stage1"]["exclusion_counts"]["code_primary"],
                1,
            )
            fit = list(module.common.read_jsonl(root / "prepared" / "fit.jsonl"))
            self.assertEqual([row["source_stage"] for row in fit], ["stage1", "stage2"])

    def test_generic_curation_is_source_agnostic(self):
        module = _load("over_context_curation", DATASET_DIR / "over_context.py")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidates = root / "candidates.jsonl"
            curation = root / "curation.jsonl"
            _write_jsonl(
                candidates,
                [
                    {
                        "schema": "small-llm-rsft-overcontext-v1",
                        "id": "source-a",
                        "source": "future_dataset",
                        "source_stage": "train",
                        "source_index": 1,
                        "source_order": 1,
                        "skill": "GENERIC_REASONING",
                        "difficulty": "simplified_fit",
                        "problem": "Long problem",
                        "reasoning": "Long reasoning",
                        "answer": "Answer",
                        "original_serialized_tokens": 3000,
                    },
                    {
                        "schema": "small-llm-rsft-overcontext-v1",
                        "id": "source-b",
                        "source": "future_dataset",
                        "source_stage": "train",
                        "source_index": 2,
                        "source_order": 2,
                        "skill": "GENERIC_REASONING",
                        "difficulty": "simplified_fit",
                        "problem": "Unsafe candidate",
                        "reasoning": "Reasoning",
                        "answer": "Answer",
                        "original_serialized_tokens": 3000,
                    },
                ],
            )
            _write_jsonl(
                curation,
                [
                    {
                        "schema": module.CURATION_SCHEMA,
                        "id": "source-a",
                        "decision": "keep",
                        "reason": "valid reasoning example",
                    },
                    {
                        "schema": module.CURATION_SCHEMA,
                        "id": "source-b",
                        "decision": "exclude_safety",
                        "reason": "exclude before teacher traffic",
                    },
                ],
            )
            result = module.prepare_keep(
                candidates_jsonl=candidates,
                curation_jsonl=curation,
                work_dir=root / "adaptation",
            )
            self.assertEqual(result["keep_records"], 1)
            kept = list(module.common.read_jsonl(root / "adaptation" / "keep.jsonl"))
            self.assertEqual(kept[0]["source"], "future_dataset")

    def test_main_builder_prepares_and_assembles_from_frozen_base(self):
        module = _load("builder", DATASET_DIR / "build.py")
        module.PARENT_TRAIN_TARGETS = 10_000
        module.common.default_token_counter = lambda: _counter
        module.superior.common.default_token_counter = lambda: _counter
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base.jsonl"
            _write_jsonl(
                base,
                [
                    {
                        "skill": "SR_INSTRUCTION_FOLLOWING",
                        "difficulty": "clean_fit",
                        "problem": f"Base prompt {index}",
                        "reasoning": "reason reason",
                        "answer": "answer",
                    }
                    for index in range(10)
                ],
            )
            stage2 = root / "stage2.jsonl"
            _write_jsonl(
                stage2,
                [
                    {
                        "uuid": f"s2-{index}",
                        "domain": "instruction_following",
                        "input": f"Summarize note {index}.",
                        "output": "<think>reason reason reason reason</think>answer",
                    }
                    for index in range(40)
                ],
            )
            work = root / "work"
            module.prepare(
                work_dir=work,
                base_jsonl=base,
                superior_stages=("stage2",),
                superior_stage2_jsonl=stage2,
                progress_every=0,
            )
            output = root / "one-percent.jsonl"
            result = module.assemble(
                work_dir=work,
                base_jsonl=base,
                output_jsonl=output,
                percent=1.0,
            )
            self.assertEqual(result["requested_total_train_targets"], 100)
            self.assertEqual(result["requested_reasoning_train_targets"], 90)
            self.assertGreaterEqual(result["projected_reasoning_train_targets"], 90)
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
