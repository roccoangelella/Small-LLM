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
    def test_atomic_serialized_token_count_matches_real_rsft_template(self) -> None:
        try:
            import tiktoken  # noqa: F401
        except ImportError:
            self.skipTest("tiktoken is required for exact R-SFT token-count parity")
        bundle_path = RSFT_DIR / "bundle.py"
        spec = importlib.util.spec_from_file_location("superior_count_parity_bundle", bundle_path)
        self.assertIsNotNone(spec)
        assert spec is not None and spec.loader is not None
        bundle = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = bundle
        spec.loader.exec_module(bundle)

        example = bundle.schema.ReasoningExample(
            skill="SR_INSTRUCTION_FOLLOWING",
            difficulty="clean_fit",
            problem="Summarize the note in three bullets.",
            reasoning="Identify the three main ideas, then state each once.",
            answer="* First idea\n* Second idea\n* Third idea",
        )
        token_spec = bundle.tokenizer.ReasoningTokenSpec("<think>", "</think>", "<answer>")
        encoder = bundle.tokenizer.ReasoningGPT2Encoder(token_spec)
        template = bundle.GPT2ChatTemplate(
            maximum_context_tokens=10_000,
            maximum_assistant_tokens=10_000,
        )
        markers = bundle.serialization.ReasoningMarkers("<think>", "</think>", "<answer>")
        conversation = bundle.ConversationRecord.from_mapping(
            bundle.serialization.to_conversation_mapping(example, markers=markers)
        )
        actual = len(template.encode_conversation(conversation, encoder).token_ids) - 1
        counted = superior_reasoning.atomic_rsft_serialized_tokens(
            problem=example.problem,
            reasoning=example.reasoning,
            answer=example.answer,
        )
        self.assertEqual(counted, actual)

    def test_instruction_filter_excludes_primary_math_and_code_but_not_format_counts(self) -> None:
        self.assertIsNone(
            superior_reasoning.instruction_exclusion_reason(
                "Summarize this report in exactly 5 bullet points and mention revenue growth of 13%."
            )
        )
        self.assertEqual(
            superior_reasoning.instruction_exclusion_reason(
                "Calculate 18% of 450 and show the equation used."
            ),
            "math_primary",
        )
        self.assertEqual(
            superior_reasoning.instruction_exclusion_reason(
                "Write a Python function that parses these records."
            ),
            "code_primary",
        )

    def test_clean_instruction_selection_is_instruction_only_and_context_safe(self) -> None:
        rows = [
            _row(0, "science", 1, prompt="Explain photosynthesis briefly."),
            _row(1, "instruction_following", 1, prompt="Summarize this note in 3 bullets."),
            _row(2, "instruction_following", 1, prompt="Solve the equation 2 + 2 = 4."),
            _row(3, "instruction_following", 1, prompt="Write a JavaScript function for this task."),
            _row(4, "instruction_following", 40, prompt="Give a concise plain-English explanation."),
            _row(5, "instruction_following", 1, prompt="Summarize   this duplicate."),
            _row(6, "instruction_following", 1, prompt="summarize this duplicate."),
            _row(7, "instruction_following", 1, prompt="Explain the literal marker <answer> briefly."),
        ]
        selected, report = superior_reasoning.select_clean_instruction(
            rows,
            context_length=20,
            token_counter=lambda text: len(text.split()),
        )
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(item["domain"] == "instruction_following" for item in selected))
        self.assertEqual(report.duplicate_input_count, 1)
        self.assertEqual(report.exclusion_counts["math_primary"], 1)
        self.assertEqual(report.exclusion_counts["code_primary"], 1)
        self.assertEqual(report.exclusion_counts["over_context"], 1)
        self.assertEqual(report.exclusion_counts["reserved_marker_collision"], 1)
        self.assertTrue(all(item["serialized_token_count"] <= 20 for item in selected))

    def test_simplification_batch_is_capped_at_four_and_uses_strict_contract(self) -> None:
        records = [
            {
                "id": f"item-{index}",
                "skill": "SR_SCIENCE",
                "problem": f"problem {index}",
                "reasoning": f"reasoning {index}",
                "answer": f"answer {index}",
            }
            for index in range(4)
        ]
        system_message, user_message = superior_reasoning.build_simplification_messages(records)
        self.assertEqual(system_message["role"], "system")
        self.assertIn("2,048-token context window", system_message["content"])
        payload = json.loads(user_message["content"])
        self.assertEqual([item["id"] for item in payload], [f"item-{index}" for index in range(4)])
        self.assertTrue(all(item["skill"] == "SR_SCIENCE" for item in payload))

        with self.assertRaisesRegex(ValueError, "cannot exceed 4"):
            superior_reasoning.build_simplification_messages([*records, records[0]])

    def test_parse_simplification_response_is_strict_and_ordered(self) -> None:
        payload = [
            {"id": "a", "problem": "P1", "reasoning": "R1", "answer": "A1"},
            {"id": "b", "problem": "P2", "reasoning": "R2", "answer": "A2"},
        ]
        parsed = superior_reasoning.parse_simplification_response(
            json.dumps(payload),
            expected_ids=("a", "b"),
        )
        self.assertEqual(tuple(item["id"] for item in parsed), ("a", "b"))

        with self.assertRaisesRegex(ValueError, "strict JSON"):
            superior_reasoning.parse_simplification_response(
                "```json\n" + json.dumps(payload) + "\n```",
                expected_ids=("a", "b"),
            )
        with self.assertRaisesRegex(ValueError, "does not match"):
            superior_reasoning.parse_simplification_response(
                json.dumps(list(reversed(payload))),
                expected_ids=("a", "b"),
            )

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
