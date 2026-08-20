from __future__ import annotations

import unittest

from post_training.rsft_prompt_wrapper_eval import (
    _selected_cases,
    _wrapper_prompt_ids,
    _wrapper_summary,
)


class FakeEncoder:
    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]


class FakeCase:
    prompt = "Who is older?"


class PromptWrapperEvalTests(unittest.TestCase):
    def test_full_suite_selects_two_cases_per_reasoning_skill(self) -> None:
        cases = _selected_cases("full")
        self.assertEqual(len(cases), 14)
        counts: dict[str, int] = {}
        for case in cases:
            counts[case.skill] = counts.get(case.skill, 0) + 1
        self.assertEqual(set(counts.values()), {2})

    def test_fast_suite_selects_one_case_per_reasoning_skill(self) -> None:
        cases = _selected_cases("fast")
        self.assertEqual(len(cases), 7)
        self.assertEqual(len({case.skill for case in cases}), 7)

    def test_question_answer_wrapper_is_raw_text_without_chat_prefix(self) -> None:
        encoder = FakeEncoder()
        token_ids = _wrapper_prompt_ids(
            "question_answer",
            case=FakeCase(),
            encoder=encoder,
            max_seq_len=2048,
        )
        text = "".join(chr(token) for token in token_ids)
        self.assertEqual(text, "Question: Who is older?\nAnswer:")
        self.assertNotIn("Assistant:", text)

    def test_plain_wrapper_is_only_problem_plus_newline(self) -> None:
        encoder = FakeEncoder()
        token_ids = _wrapper_prompt_ids(
            "plain",
            case=FakeCase(),
            encoder=encoder,
            max_seq_len=2048,
        )
        text = "".join(chr(token) for token in token_ids)
        self.assertEqual(text, "Who is older?\n")

    def test_wrapper_summary_keeps_answer_and_protocol_axes_separate(self) -> None:
        protocol = {
            "well_formed": True,
            "ordered": True,
            "reasoning_start_count": 1,
            "reasoning_end_count": 1,
            "answer_start_count": 1,
            "reasoning_tokens": 3,
            "answer_tokens": 2,
            "terminated_with_eos": True,
            "runaway": False,
            "nonempty_reasoning": True,
            "nonempty_answer": True,
        }
        rows = [
            {
                "answer_correct_any_format": True,
                "strict_correct": True,
                "protocol": protocol,
            },
            {
                "answer_correct_any_format": True,
                "strict_correct": False,
                "protocol": {**protocol, "well_formed": False, "reasoning_start_count": 0},
            },
        ]
        summary = _wrapper_summary(rows)
        self.assertEqual(summary["answer_accuracy_any_format"], 1.0)
        self.assertEqual(summary["strict_protocol_answer_accuracy"], 0.5)
        self.assertEqual(summary["reasoning_start_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
