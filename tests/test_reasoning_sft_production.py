from __future__ import annotations

from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from post_training.sft.schema import TokenizedSFTRecord


REPO = Path(__file__).resolve().parents[1]
RSFT_DIR = REPO / "post_training" / "R-SFT"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load(name: str):
    module_name = f"small_llm_rsft_production_test_{name}"
    path = RSFT_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


bundle = _load("bundle")
production = _load("production")


class _Response:
    def __init__(self, content: str) -> None:
        self.content = content


class _Teacher:
    def __init__(self, *, fail_if_called: bool = False) -> None:
        self.calls = 0
        self.fail_if_called = fail_if_called

    def complete_text(self, prompt: str) -> _Response:
        if self.fail_if_called:
            raise AssertionError("completed generation should not call the teacher again")
        self.calls += 1
        return _Response(
            json.dumps(
                [
                    {
                        "problem": f"problem-{self.calls}",
                        "reasoning": f"reasoning-{self.calls}",
                        "answer": f"answer-{self.calls}",
                    }
                ]
            )
        )


def _reasoning_records(per_cell: int):
    records = []
    for skill in bundle.prompts.R0_SKILLS:
        for difficulty in bundle.generation.R0_DIFFICULTIES:
            for index in range(per_cell):
                records.append(
                    bundle.schema.ReasoningExample(
                        skill=skill,
                        difficulty=difficulty,
                        problem=f"{skill}-{difficulty}-problem-{index}",
                        reasoning=f"{skill}-{difficulty}-reasoning-{index}",
                        answer=f"{skill}-{difficulty}-answer-{index}",
                    )
                )
    return tuple(records)


def _token_record(record_id: str, source: str, target_tokens: int) -> TokenizedSFTRecord:
    token_ids = tuple([50_256] + list(range(100, 100 + target_tokens)))
    return TokenizedSFTRecord(
        record_id=record_id,
        source=source,
        split="train",
        token_ids=token_ids,
        target_mask=tuple(True for _ in range(target_tokens)),
    )


class ReasoningSFTPartitionTests(unittest.TestCase):
    def test_630_pilot_partition_keeps_every_cell_in_both_heldout_splits(self) -> None:
        records = _reasoning_records(30)
        counts = bundle.validate_reasoning_matrix(records, examples_per_cell=30)
        self.assertEqual(len(counts), 21)
        partition = bundle.partition_reasoning_records(records, heldout_per_cell=1, seed=17)
        self.assertEqual(len(partition["train"]), 21 * 28)
        self.assertEqual(len(partition["validation"]), 21)
        self.assertEqual(len(partition["test"]), 21)
        for split, expected in (("train", 28), ("validation", 1), ("test", 1)):
            per_cell = Counter((item.skill, item.difficulty) for item in partition[split])
            self.assertEqual(set(per_cell.values()), {expected})

    def test_partition_is_deterministic(self) -> None:
        records = _reasoning_records(3)
        first = bundle.partition_reasoning_records(records, heldout_per_cell=1, seed=17)
        second = bundle.partition_reasoning_records(tuple(reversed(records)), heldout_per_cell=1, seed=17)
        first_ids = {
            split: [bundle.serialization.stable_conversation_id(item) for item in values]
            for split, values in first.items()
        }
        second_ids = {
            split: [bundle.serialization.stable_conversation_id(item) for item in values]
            for split, values in second.items()
        }
        self.assertEqual(first_ids, second_ids)


class ReasoningSFTRetentionTests(unittest.TestCase):
    def test_matched_arms_use_symmetric_retention_reference(self) -> None:
        self.assertEqual(bundle.retention_target_tokens_for_matched_arms(900, 900), 100)
        self.assertEqual(
            bundle.retention_target_tokens_for_matched_arms(882, 918),
            100,
        )

    def test_retention_samples_only_requested_instruction_sources(self) -> None:
        stream = []
        for index in range(20):
            stream.extend(
                [
                    _token_record(f"a-{index}", "a", 10),
                    _token_record(f"replay-{index}", "climbmix-replay", 10),
                    _token_record(f"b-{index}", "b", 10),
                ]
            )
        selected = bundle.select_s0_retention_records(
            stream,
            source_shares={"a": 0.75, "b": 0.25},
            target_tokens=100,
        )
        self.assertTrue(selected)
        self.assertEqual({record.source for record in selected}, {"a", "b"})
        tokens = Counter()
        for record in selected:
            tokens[record.source] += record.target_token_count
        self.assertGreaterEqual(tokens["a"], 75)
        self.assertGreaterEqual(tokens["b"], 25)


class ReasoningSFTResumableGenerationTests(unittest.TestCase):
    def test_generation_freezes_and_completed_rerun_makes_no_teacher_calls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            teacher = _Teacher()
            first = production.generate_resumable(
                root,
                examples_per_cell=1,
                batch_size=1,
                seed=17,
                client=teacher,
            )
            self.assertEqual(first["records"], 21)
            self.assertEqual(first["calls"], 21)
            self.assertEqual(teacher.calls, 21)
            self.assertTrue((root / "reasoning.jsonl").is_file())
            self.assertTrue((root / "generation-manifest.json").is_file())
            self.assertEqual(len(list((root / "batches").glob("*.json"))), 21)

            second = production.generate_resumable(
                root,
                examples_per_cell=1,
                batch_size=1,
                seed=17,
                client=_Teacher(fail_if_called=True),
            )
            self.assertTrue(second["resumed_complete"])
            self.assertEqual(second["records"], 21)

    def test_saved_batch_plan_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            production.generate_resumable(
                root,
                examples_per_cell=1,
                batch_size=1,
                seed=17,
                client=_Teacher(),
            )
            (root / "reasoning.jsonl").unlink()
            with self.assertRaisesRegex(RuntimeError, "generation plan"):
                production.generate_resumable(
                    root,
                    examples_per_cell=2,
                    batch_size=1,
                    seed=17,
                    client=_Teacher(),
                )


if __name__ == "__main__":
    unittest.main()
