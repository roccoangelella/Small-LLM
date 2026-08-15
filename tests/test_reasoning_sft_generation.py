from __future__ import annotations

from collections import Counter
import importlib.util
import json
from pathlib import Path
import re
import sys
import unittest


RSFT_DIR = Path(__file__).resolve().parents[1] / "post_training" / "R-SFT"


def _load(name: str):
    module_name = f"small_llm_rsft_test_{name}"
    path = RSFT_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


schema = _load("schema")
generate = _load("generate")
serialization = _load("serialization")
mixture = _load("mixture")


class _Response:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeTeacher:
    def __init__(self, *, malformed: bool = False) -> None:
        self.prompts: list[str] = []
        self.counter = 0
        self.malformed = malformed

    def complete_text(self, prompt: str) -> _Response:
        self.prompts.append(prompt)
        if self.malformed:
            return _Response('heading\n[{"problem":"p","reasoning":"r","answer":"a"}]')
        match = re.search(r"Generate (\d+) self-contained", prompt)
        if match is None:
            raise AssertionError("prompt does not expose requested batch size")
        count = int(match.group(1))
        records = []
        for _ in range(count):
            self.counter += 1
            records.append(
                {
                    "problem": f"Problem {self.counter}",
                    "reasoning": f"Reasoning {self.counter}",
                    "answer": f"Answer {self.counter}",
                }
            )
        return _Response(json.dumps(records))


class ReasoningSFTSchemaTests(unittest.TestCase):
    def test_teacher_batch_accepts_only_exact_three_field_objects(self) -> None:
        records = schema.parse_teacher_batch(
            '[{"problem":"P","reasoning":"R","answer":"A"}]',
            expected_count=1,
        )
        self.assertEqual(records[0].problem, "P")

        with self.assertRaisesRegex(ValueError, "exactly problem/reasoning/answer"):
            schema.parse_teacher_batch(
                '[{"problem":"P","reasoning":"R","answer":"A","model":"gemini"}]',
                expected_count=1,
            )

    def test_teacher_batch_rejects_surrounding_commentary_and_wrong_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "valid JSON"):
            schema.parse_teacher_batch(
                'Generated results:\n[{"problem":"P","reasoning":"R","answer":"A"}]',
                expected_count=1,
            )
        with self.assertRaisesRegex(ValueError, "expected 2"):
            schema.parse_teacher_batch(
                '[{"problem":"P","reasoning":"R","answer":"A"}]',
                expected_count=2,
            )


class ReasoningSFTGenerationTests(unittest.TestCase):
    def test_uniform_generator_covers_every_r0_cell_and_hides_level_labels(self) -> None:
        teacher = _FakeTeacher()
        records = generate.generate_uniform_dataset(
            teacher,
            examples_per_cell=3,
            batch_size=2,
            seed=17,
        )
        expected_cells = {
            (skill, difficulty)
            for skill in generate.prompts.R0_SKILLS
            for difficulty in generate.R0_DIFFICULTIES
        }
        counts = Counter((record.skill, record.difficulty) for record in records)
        self.assertEqual(set(counts), expected_cells)
        self.assertTrue(all(count == 3 for count in counts.values()))
        self.assertEqual(len(records), 7 * 3 * 3)
        self.assertTrue(teacher.prompts)
        for prompt in teacher.prompts:
            self.assertNotIn("L1", prompt)
            self.assertNotIn("L2", prompt)
            self.assertNotIn("L3", prompt)

    def test_uniform_generator_fails_closed_on_malformed_teacher_json(self) -> None:
        with self.assertRaisesRegex(ValueError, "valid JSON"):
            generate.generate_uniform_dataset(
                _FakeTeacher(malformed=True),
                examples_per_cell=1,
                batch_size=1,
            )

    def test_plan_batches_each_cell_without_changing_uniform_target(self) -> None:
        plan = generate.build_uniform_generation_plan(examples_per_cell=23, batch_size=10)
        summary = generate.plan_summary(plan)
        self.assertEqual(summary["total_examples"], 7 * 3 * 23)
        self.assertEqual(summary["total_calls"], 7 * 3 * 3)
        self.assertTrue(all(value == 23 for value in summary["cells"].values()))


class ReasoningSFTSerializationTests(unittest.TestCase):
    def test_reasoning_serialization_keeps_markers_configurable(self) -> None:
        example = schema.ReasoningExample(
            skill="DED",
            difficulty="L2",
            problem="What follows?",
            reasoning="First infer B, then C.",
            answer="C follows.",
        )
        markers = serialization.ReasoningMarkers(
            reasoning_start="<R>",
            reasoning_end="</R>",
            answer_start="<A>",
        )
        record = serialization.to_conversation_mapping(example, markers=markers)
        self.assertEqual(record["source"], "r0-reasoning")
        self.assertEqual(record["messages"][0], {"role": "user", "content": "What follows?"})
        self.assertEqual(
            record["messages"][1]["content"],
            "<R>First infer B, then C.</R><A>C follows.",
        )
        self.assertEqual(record["metadata"]["skill"], "DED")
        self.assertEqual(record["metadata"]["difficulty"], "L2")
        self.assertEqual(
            serialization.stable_conversation_id(example),
            serialization.stable_conversation_id(example),
        )


class ReasoningSFTRetentionMixtureTests(unittest.TestCase):
    def test_retention_is_ten_percent_of_target_token_source_shares(self) -> None:
        shares = mixture.build_rsft_source_shares(
            {"instructions-a": 0.75, "instructions-b": 0.25}
        )
        self.assertAlmostEqual(shares["r0-reasoning"], 0.90)
        self.assertAlmostEqual(shares["instructions-a"], 0.075)
        self.assertAlmostEqual(shares["instructions-b"], 0.025)
        self.assertAlmostEqual(sum(shares.values()), 1.0)

    def test_retention_submixture_must_be_normalized(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum to one"):
            mixture.build_rsft_source_shares({"instructions": 0.9})


if __name__ == "__main__":
    unittest.main()
