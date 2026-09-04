import json
import unittest

from trainer.base_prompt_judge import (
    JUDGE_SYSTEM_PROMPT,
    build_messages,
    judge_bundle,
    parse_judgments,
)
from trainer.pretraining_eval_v2 import _summary


class FakeJudgeClient:
    model = "fake-gemini"

    def complete(self, messages):
        payload = json.loads(messages[1]["content"])
        rows = []
        for index, item in enumerate(payload):
            verdict = "correct" if index % 2 == 0 else "incorrect"
            rows.append(
                {
                    "id": item["id"],
                    "verdict": verdict,
                    "reason": "semantic test verdict",
                }
            )
        return {
            "content": json.dumps(rows),
            "model": self.model,
            "finish_reason": "stop",
        }


def _prompt_suite():
    cases = [
        {
            "name": "factual_00",
            "family": "factual",
            "qualitative": False,
            "scored": True,
            "judge_status": "pending",
            "prompt": "Question: At what Celsius temperature does water freeze?\nAnswer:",
            "reference_answer": "0",
            "continuation": " At a temperature of -40 C.",
            "response_tokens": 7,
        },
        {
            "name": "factual_01",
            "family": "factual",
            "qualitative": False,
            "scored": True,
            "judge_status": "pending",
            "prompt": "Question: What is the capital of France?\nAnswer:",
            "reference_answer": "Paris",
            "continuation": " Paris is the capital.",
            "response_tokens": 5,
        },
        {
            "name": "qualitative_00",
            "family": "qualitative",
            "qualitative": True,
            "scored": False,
            "judge_status": "not_applicable",
            "prompt": "The rain stopped and the street was ",
            "reference_answer": None,
            "continuation": " quiet.",
            "response_tokens": 2,
        },
    ]
    return {
        "schema": "small-llm-pretraining-base-prompts-v2",
        "suite_identity": {
            "prompt_set_id": "base-prompt-v2-unique-120-2026-09-04"
        },
        "scoring_contract": {
            "local_string_or_regex_scoring": False,
            "status": "pending",
        },
        "greedy": {"cases": list(cases)},
        "sampled": {"cases": list(cases)},
    }


class BasePromptRawEvidenceTests(unittest.TestCase):
    def test_raw_summary_has_no_local_accuracy(self):
        rows = [
            {
                "scored": True,
                "qualitative": False,
                "family": "factual",
                "response_tokens": 3,
            }
        ]
        summary = _summary(rows)
        self.assertIsNone(summary["accuracy"])
        self.assertEqual(summary["judge_status"], "pending")
        self.assertNotIn("passed", rows[0])
        self.assertNotIn("checks", rows[0])

    def test_judge_prompt_explicitly_rejects_substring_false_positive(self):
        self.assertIn('reference_answer "0" does not make "-40 C" correct', JUDGE_SYSTEM_PROMPT)
        self.assertIn("semantic correctness, not literal substring overlap", JUDGE_SYSTEM_PROMPT)


class BasePromptJudgeContractTests(unittest.TestCase):
    def test_build_messages_carries_reference_and_continuation(self):
        rows = [
            {
                "id": "case-1",
                "family": "factual",
                "prompt": "Q",
                "reference_answer": "A",
                "continuation": "B",
            }
        ]
        system, user = build_messages(rows)
        self.assertEqual(system["role"], "system")
        payload = json.loads(user["content"])
        self.assertEqual(payload[0]["reference_answer"], "A")
        self.assertEqual(payload[0]["continuation"], "B")

    def test_parse_judgments_is_fail_closed(self):
        with self.assertRaises(ValueError):
            parse_judgments(
                '[{"id":"case-1","verdict":"maybe","reason":"x"}]',
                expected_ids=["case-1"],
            )
        with self.assertRaises(ValueError):
            parse_judgments(
                '[{"id":"wrong","verdict":"correct","reason":"x"}]',
                expected_ids=["case-1"],
            )

    def test_judge_bundle_scores_only_objective_cases(self):
        bundle = {
            "schema": "small-llm-pretraining-evaluation-v2",
            "base_prompt_suite_v2": _prompt_suite(),
        }
        result = judge_bundle(
            bundle,
            client=FakeJudgeClient(),
            source_sha256="abc",
            batch_size=2,
            max_attempts=1,
            retry_delay_seconds=0,
        )
        target = result["targets"]["pretraining"]
        self.assertEqual(target["greedy"]["summary"]["judged_cases"], 2)
        self.assertEqual(target["sampled"]["summary"]["judged_cases"], 2)
        self.assertEqual(target["greedy"]["summary"]["accuracy"], 0.5)
        judged_names = [row["name"] for row in target["greedy"]["cases"]]
        self.assertNotIn("qualitative_00", judged_names)


if __name__ == "__main__":
    unittest.main()
