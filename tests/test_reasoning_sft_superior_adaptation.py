from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "post_training" / "R-SFT" / "dataset" / "adapt_superior_reasoning.py"
SPEC = importlib.util.spec_from_file_location("small_llm_rsft_superior_adaptation_test", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
adapt = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = adapt
SPEC.loader.exec_module(adapt)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_row(index: int, problem: str, reasoning: str = "reason briefly", answer: str = "answer") -> dict[str, str]:
    return {
        "uuid": f"source-{index}",
        "domain": "instruction_following",
        "input": problem,
        "output": f"<think>{reasoning}</think>{answer}",
    }


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content
        self.model = "gemini-3.7-flash"
        self.finish_reason = "stop"
        self.usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}


class _FakeClient:
    def __init__(self) -> None:
        self.calls = 0
        self.system_prompts: list[str] = []

    def complete(self, messages):
        self.calls += 1
        self.system_prompts.append(messages[0]["content"])
        payload = json.loads(messages[1]["content"])
        return _FakeResponse(
            json.dumps(
                [
                    {
                        "id": item["id"],
                        "problem": f"shortened {item['id']} task",
                        "reasoning": "Identify the request and satisfy its constraints directly.",
                        "answer": "A concise response that satisfies the rewritten request.",
                    }
                    for item in payload
                ]
            )
        )


class SuperiorReasoningAdaptationTests(unittest.TestCase):
    def test_variant_d_pipeline_prepares_adapts_resumes_and_finalizes(self) -> None:
        rows = [
            _source_row(0, "give three concise travel tips"),
            _source_row(1, "write a long but ordinary instruction response", reasoning="word " * 2100),
            _source_row(2, "explain calculus and derive an equation"),
            _source_row(3, "write python code for a parser"),
            _source_row(4, "explain why the literal marker <think> is reserved"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_jsonl = root / "baseline.jsonl"
            baseline_records = [
                {
                    "skill": "SR_INSTRUCTION_FOLLOWING",
                    "difficulty": "clean_fit",
                    "problem": "give three concise travel tips",
                    "reasoning": "reason briefly",
                    "answer": "answer",
                },
                {
                    "skill": "DED",
                    "difficulty": "L1",
                    "problem": "If A implies B and A holds, what follows?",
                    "reasoning": "Apply modus ponens.",
                    "answer": "B follows.",
                },
            ]
            baseline_jsonl.write_text(
                "".join(json.dumps(row) + "\n" for row in baseline_records), encoding="utf-8"
            )
            baseline_manifest = root / "baseline.manifest.json"
            baseline_manifest.write_text(
                json.dumps(
                    {
                        "schema": "small-llm-superior-reasoning-production-v1",
                        "policy": adapt.superior.PRODUCTION_FILTER_VERSION,
                        "context_length": 2048,
                        "source_rows": 5,
                        "domain_counts": {"instruction_following": 5},
                        "valid_unique_instruction_rows": 5,
                        "rejected_output_count": 0,
                        "duplicate_input_count": 0,
                        "exclusion_counts": {
                            "code_primary": 1,
                            "math_primary": 1,
                            "over_context": 1,
                            "reserved_marker_collision": 1,
                        },
                        "selected_count": 1,
                        "gemini_rows": 1,
                        "combined_rows": 2,
                        "output_sha256": _sha256(baseline_jsonl),
                    }
                ),
                encoding="utf-8",
            )

            work = root / "work"
            prepared = adapt.prepare_candidates(
                work,
                baseline_manifest=baseline_manifest,
                rows=rows,
                token_counter=lambda text: len(text.split()),
                progress_every=0,
            )
            self.assertEqual(prepared["records"], 1)
            candidate = next(adapt._read_jsonl(work / "candidates.jsonl"))
            self.assertEqual(candidate["id"], "source-1")
            self.assertGreater(candidate["original_serialized_tokens"], 2048)

            client = _FakeClient()
            result = adapt.adapt_candidates(work, client=client, retry_delay_seconds=0)
            self.assertEqual(result["records"], 1)
            self.assertEqual(client.calls, 1)
            self.assertEqual(client.system_prompts, [adapt.superior.SIMPLIFICATION_SYSTEM_PROMPT])

            resumed = adapt.adapt_candidates(work, client=client, retry_delay_seconds=0)
            self.assertTrue(resumed["resumed_complete"])
            self.assertEqual(client.calls, 1)

            curation = work / "manual-curation.jsonl"
            curation.write_text(
                json.dumps(
                    {
                        "schema": adapt.MANUAL_CURATION_SCHEMA,
                        "id": "source-1",
                        "decision": "keep",
                        "reason": "Unit-test instruction task; no primary math or programming.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            final_path = root / "complete.jsonl"
            final = adapt.finalize_complete_dataset(
                work,
                baseline_jsonl=baseline_jsonl,
                baseline_manifest=baseline_manifest,
                manual_curation_jsonl=curation,
                output_jsonl=final_path,
            )
            self.assertEqual(final["unchanged_superior_rows"], 1)
            self.assertEqual(final["adapted_superior_rows"], 1)
            self.assertEqual(final["gemini_rows"], 1)
            self.assertEqual(final["combined_rows"], 3)
            final_rows = list(adapt._read_jsonl(final_path))
            self.assertEqual(
                sorted(row["difficulty"] for row in final_rows),
                ["L1", "clean_fit", "simplified_fit"],
            )
            for row in final_rows:
                self.assertLessEqual(
                    adapt.superior.atomic_rsft_serialized_tokens(
                        problem=row["problem"], reasoning=row["reasoning"], answer=row["answer"]
                    ),
                    2048,
                )

    def test_checkpoint_finalizer_uses_only_available_kept_batches(self) -> None:
        rows = [
            _source_row(0, "give three concise travel tips"),
            _source_row(1, "write a long ordinary instruction response one", reasoning="word " * 2100),
            _source_row(2, "write a long ordinary instruction response two", reasoning="word " * 2100),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_jsonl = root / "baseline.jsonl"
            baseline_records = [
                {
                    "skill": "SR_INSTRUCTION_FOLLOWING",
                    "difficulty": "clean_fit",
                    "problem": "give three concise travel tips",
                    "reasoning": "reason briefly",
                    "answer": "answer",
                },
                {
                    "skill": "DED",
                    "difficulty": "L1",
                    "problem": "If A implies B and A holds, what follows?",
                    "reasoning": "Apply modus ponens.",
                    "answer": "B follows.",
                },
            ]
            baseline_jsonl.write_text(
                "".join(json.dumps(row) + "\n" for row in baseline_records), encoding="utf-8"
            )
            baseline_manifest = root / "baseline.manifest.json"
            baseline_manifest.write_text(
                json.dumps(
                    {
                        "schema": "small-llm-superior-reasoning-production-v1",
                        "policy": adapt.superior.PRODUCTION_FILTER_VERSION,
                        "context_length": 2048,
                        "source_rows": 3,
                        "domain_counts": {"instruction_following": 3},
                        "valid_unique_instruction_rows": 3,
                        "rejected_output_count": 0,
                        "duplicate_input_count": 0,
                        "exclusion_counts": {"over_context": 2},
                        "selected_count": 1,
                        "gemini_rows": 1,
                        "combined_rows": 2,
                        "output_sha256": _sha256(baseline_jsonl),
                    }
                ),
                encoding="utf-8",
            )

            work = root / "work"
            adapt.prepare_candidates(
                work,
                baseline_manifest=baseline_manifest,
                rows=rows,
                token_counter=lambda text: len(text.split()),
                progress_every=0,
            )
            client = _FakeClient()
            adapted = adapt.adapt_candidates(
                work,
                batch_size=1,
                max_attempts=1,
                retry_delay_seconds=0,
                request_interval_seconds=0,
                client=client,
                max_batches=1,
            )
            self.assertFalse(adapted["complete"])
            self.assertEqual(client.calls, 1)

            curation = work / "manual-curation.jsonl"
            curation.write_text(
                "".join(
                    json.dumps(
                        {
                            "schema": adapt.MANUAL_CURATION_SCHEMA,
                            "id": f"source-{index}",
                            "decision": "keep",
                            "reason": "Instruction task retained for checkpoint test.",
                        }
                    )
                    + "\n"
                    for index in (1, 2)
                ),
                encoding="utf-8",
            )
            output = root / "checkpoint.jsonl"
            result = adapt.finalize_checkpoint_dataset(
                work,
                baseline_jsonl=baseline_jsonl,
                baseline_manifest=baseline_manifest,
                manual_curation_jsonl=curation,
                output_jsonl=output,
                batch_size=1,
            )
            self.assertEqual(result["schema"], adapt.CHECKPOINT_SCHEMA)
            self.assertEqual(result["accepted_batches"], 1)
            self.assertEqual(result["accepted_adapted_records"], 1)
            self.assertEqual(result["adapted_superior_rows"], 1)
            self.assertEqual(result["pending_kept_adaptation_rows"], 1)
            self.assertEqual(result["combined_rows"], 3)
            self.assertEqual(len(list(adapt._read_jsonl(output))), 3)

    def test_rejected_attempts_do_not_permanently_exhaust_a_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "attempts").mkdir()
            ids = ["source-9"]
            for attempt in (1, 2, 3):
                (root / "attempts" / f"batch-00007-attempt-{attempt:02d}.json").write_text(
                    json.dumps(
                        {
                            "schema": adapt.ATTEMPT_SCHEMA,
                            "batch_index": 7,
                            "attempt": attempt,
                            "ids": ids,
                            "status": "rejected",
                            "error_type": "RuntimeError",
                            "error": "temporary outage",
                        }
                    ),
                    encoding="utf-8",
                )
            highest, recovered = adapt._existing_attempt_state(
                root,
                batch_index=7,
                expected_ids=ids,
            )
            self.assertEqual(highest, 3)
            self.assertIsNone(recovered)
            self.assertEqual(adapt._attempt_path(root, 7, highest + 1).name, "batch-00007-attempt-04.json")

    def test_batch_size_cannot_exceed_selected_variant_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_size"):
            adapt.adapt_candidates("/tmp/does-not-matter", batch_size=5)


if __name__ == "__main__":
    unittest.main()
