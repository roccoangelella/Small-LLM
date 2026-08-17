from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "post_training" / "R-SFT" / "prompts.py"
SPEC = importlib.util.spec_from_file_location("reasoning_sft_prompts", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
prompts = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prompts
SPEC.loader.exec_module(prompts)


class ReasoningSFTPromptTests(unittest.TestCase):
    def test_all_r0_skills_have_prompt_specs(self) -> None:
        self.assertEqual(
            set(prompts.R0_SKILLS),
            {"INF", "DED", "REL", "CSP", "IND", "ABD", "MAG"},
        )
        self.assertEqual(set(prompts.SKILL_PROMPT_SPECS), set(prompts.R0_SKILLS))

    def test_every_prompt_has_shared_generation_contract(self) -> None:
        for skill in prompts.R0_SKILLS:
            with self.subTest(skill=skill):
                prompt = prompts.build_generation_prompt(skill)
                self.assertIn("Generate 10 self-contained", prompt)
                self.assertIn("fully self-contained", prompt)
                self.assertIn("mutually consistent", prompt)
                self.assertIn("Prefer open-ended questions", prompt)
                self.assertIn("Do not force answers into a yes/no format", prompt)
                self.assertIn("Here is one example", prompt)
                self.assertIn('"problem": "..."', prompt)
                self.assertIn('"reasoning": "..."', prompt)
                self.assertIn('"answer": "..."', prompt)
                self.assertIn("exactly 10 objects", prompt)
                self.assertIn("Do not include markdown fences", prompt)

    def test_internal_skill_code_is_not_emitted_as_teacher_label(self) -> None:
        for skill in prompts.R0_SKILLS:
            with self.subTest(skill=skill):
                prompt = prompts.build_generation_prompt(skill)
                self.assertNotIn(f"Skill: {skill}", prompt)
                self.assertNotIn(f"skill = {skill}", prompt)

    def test_structural_requirements_are_passed_as_plain_language(self) -> None:
        requirement = (
            "Require several dependent relations to be combined, with one small elimination step."
        )
        prompt = prompts.build_generation_prompt(
            "REL",
            batch_size=7,
            structural_requirements=requirement,
        )
        self.assertIn(requirement, prompt)
        self.assertIn("exactly 7 objects", prompt)
        self.assertNotIn("L1", prompt)
        self.assertNotIn("L2", prompt)
        self.assertNotIn("L3", prompt)

    def test_invalid_skill_and_batch_size_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown R0 reasoning skill"):
            prompts.build_generation_prompt("MATH")
        for invalid in (0, -1, True, 1.5):
            with self.subTest(batch_size=invalid):
                with self.assertRaisesRegex(ValueError, "batch_size"):
                    prompts.build_generation_prompt("DED", batch_size=invalid)

    def test_skill_prompts_are_distinct(self) -> None:
        rendered = {skill: prompts.build_generation_prompt(skill) for skill in prompts.R0_SKILLS}
        self.assertEqual(len(set(rendered.values())), len(prompts.R0_SKILLS))
        self.assertIn("immediate logical inference", rendered["INF"])
        self.assertIn("must be deduction", rendered["DED"])
        self.assertIn("relational reasoning", rendered["REL"])
        self.assertIn("satisfying several explicit constraints", rendered["CSP"])
        self.assertIn("induction from controlled observations", rendered["IND"])
        self.assertIn("controlled abduction", rendered["ABD"])
        self.assertIn("numerical magnitude", rendered["MAG"])


if __name__ == "__main__":
    unittest.main()
