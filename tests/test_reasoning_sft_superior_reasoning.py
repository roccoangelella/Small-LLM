from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


RSFT_DIR = Path(__file__).resolve().parents[1] / "post_training" / "R-SFT"
MODULE_PATH = RSFT_DIR / "dataset" / "superior_reasoning.py"
SPEC = importlib.util.spec_from_file_location("small_llm_rsft_superior_reasoning_test", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
superior_reasoning = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = superior_reasoning
SPEC.loader.exec_module(superior_reasoning)


def _row(
    index: int,
    domain: str,
    reasoning_tokens: int,
    *,
    prompt: str | None = None,
    output: str | None = None,
) -> dict[str, str]:
    return {
        "uuid": f"id-{index}",
        "input": prompt or f"prompt {index}",
        "output": output or f'<think>{"x " * reasoning_tokens}</think>answer {index}',
        "domain": domain,
        "meta": "{}",
    }


class SuperiorReasoningDatasetTests(unittest.TestCase):
    def test_parse_teacher_output_splits_reasoning_and_answer(self) -> None:
        parsed = superior_reasoning.parse_teacher_output(
            "<think>short reasoning</think><answer>final answer</answer>"
        )
        self.assertEqual(parsed.reasoning, "short reasoning")
        self.assertEqual(parsed.answer, "final answer")

        with self.assertRaisesRegex(ValueError, "missing </think>"):
            superior_reasoning.parse_teacher_output("<think>broken")

    def test_select_shortest_respects_requested_stratification(self) -> None:
        rows = [
            _row(0, "science", 5),
            _row(1, "science", 2),
            _row(2, "science", 8),
            _row(3, "instruction_following", 7),
            _row(4, "instruction_following", 1),
            _row(5, "instruction_following", 4),
            _row(6, "instruction_following", 3),
        ]
        selected, report = superior_reasoning.select_shortest(
            rows,
            total=5,
            science_share=0.4,
            token_counter=lambda text: len(text.split()),
        )

        self.assertEqual(report.selected_counts, {"science": 2, "instruction_following": 3})
        science_lengths = sorted(
            record["reasoning_token_count"] for record in selected if record["domain"] == "science"
        )
        instruction_lengths = sorted(
            record["reasoning_token_count"]
            for record in selected
            if record["domain"] == "instruction_following"
        )
        self.assertEqual(science_lengths, [2, 5])
        self.assertEqual(instruction_lengths, [1, 3, 4])

    def test_select_shortest_backfills_when_science_pool_is_short(self) -> None:
        rows = [
            _row(0, "science", 1),
            *[_row(index, "instruction_following", index) for index in range(1, 7)],
        ]
        _selected, report = superior_reasoning.select_shortest(
            rows,
            total=5,
            science_share=0.4,
            token_counter=lambda text: len(text.split()),
        )
        self.assertEqual(report.selected_counts, {"science": 1, "instruction_following": 4})
        self.assertEqual(report.total_selected, 5)

    def test_exact_duplicate_prompts_and_malformed_outputs_are_rejected(self) -> None:
        rows = [
            _row(0, "science", 4, prompt="Same   Prompt"),
            _row(1, "science", 2, prompt="same prompt"),
            _row(2, "instruction_following", 3, output="no reasoning markers"),
            _row(3, "instruction_following", 1),
        ]
        selected, report = superior_reasoning.select_shortest(
            rows,
            total=2,
            science_share=0.5,
            token_counter=lambda text: len(text.split()),
        )
        self.assertEqual(report.duplicate_input_count, 1)
        self.assertEqual(report.rejected_output_count, 1)
        self.assertEqual(len(selected), 2)

    def test_write_combined_jsonl_merges_existing_gemini_records(self) -> None:
        superior = [
            {
                "domain": "science",
                "problem": "P",
                "reasoning": "R",
                "answer": "A",
                "source_id": "id-1",
                "source_index": 1,
                "reasoning_token_count": 1,
            }
        ]
        gemini = {
            "skill": "DED",
            "difficulty": "L1",
            "problem": "GP",
            "reasoning": "GR",
            "answer": "GA",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gemini_path = root / "gemini.jsonl"
            gemini_path.write_text(json.dumps(gemini) + "\n", encoding="utf-8")
            output = root / "combined.jsonl"
            superior_reasoning.write_combined_jsonl(
                superior,
                output,
                gemini_jsonl=gemini_path,
                seed=17,
            )
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(records), 2)
        self.assertEqual({record["skill"] for record in records}, {"SR_SCIENCE", "DED"})
        self.assertTrue(all(set(record) == {"skill", "difficulty", "problem", "reasoning", "answer"} for record in records))


if __name__ == "__main__":
    unittest.main()
