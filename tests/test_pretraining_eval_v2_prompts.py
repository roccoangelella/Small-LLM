from collections import Counter
import unittest

from trainer.pretraining_eval_v2 import BASE_PROMPT_CASES_V2, BASE_PROMPT_SET_ID


class BasePromptV2DefinitionTests(unittest.TestCase):
    def test_full_suite_has_120_unique_ids_and_prompts(self) -> None:
        self.assertEqual(len(BASE_PROMPT_CASES_V2), 120)
        names = [case.name for case in BASE_PROMPT_CASES_V2]
        prompts = [case.prompt for case in BASE_PROMPT_CASES_V2]
        self.assertEqual(len(set(names)), 120)
        self.assertEqual(len(set(prompts)), 120)

    def test_full_suite_has_expected_scored_and_qualitative_counts(self) -> None:
        scored = [case for case in BASE_PROMPT_CASES_V2 if not case.qualitative]
        qualitative = [case for case in BASE_PROMPT_CASES_V2 if case.qualitative]
        self.assertEqual(len(scored), 100)
        self.assertEqual(len(qualitative), 20)
        self.assertTrue(all(case.family == "qualitative" for case in qualitative))

    def test_each_scored_family_has_20_unique_prompts(self) -> None:
        scored = [case for case in BASE_PROMPT_CASES_V2 if not case.qualitative]
        counts = Counter(case.family for case in scored)
        self.assertEqual(
            counts,
            Counter(
                {
                    "factual": 20,
                    "arithmetic": 20,
                    "extraction": 20,
                    "classification": 20,
                    "transformation": 20,
                }
            ),
        )
        for family in counts:
            family_prompts = [case.prompt for case in scored if case.family == family]
            self.assertEqual(len(set(family_prompts)), 20)

    def test_prompt_set_identity_marks_unique_120_revision(self) -> None:
        self.assertEqual(BASE_PROMPT_SET_ID, "base-prompt-v2-unique-120-2026-09-04")


if __name__ == "__main__":
    unittest.main()
