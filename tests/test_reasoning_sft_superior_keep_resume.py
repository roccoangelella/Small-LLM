from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
P = REPO / "post_training" / "R-SFT" / "dataset" / "resume_superior_keep_adaptation.py"
S = importlib.util.spec_from_file_location("keep_resume_test", P)
assert S and S.loader
resume = importlib.util.module_from_spec(S)
sys.modules[S.name] = resume
S.loader.exec_module(resume)
base = resume.base


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def src(i, p, r="word " * 2100):
    return {
        "uuid": f"source-{i}",
        "domain": "instruction_following",
        "input": p,
        "output": f"<think>{r}</think>answer",
    }


class Response:
    def __init__(self, c):
        self.content = c
        self.model = "gemini-3.7-flash"
        self.finish_reason = "stop"
        self.usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


class Client:
    def __init__(self, *a, **k):
        pass

    def complete(self, messages):
        rows = json.loads(messages[1]["content"])
        return Response(
            json.dumps(
                [
                    {
                        "id": r["id"],
                        "problem": f"unique shortened {r['id']}",
                        "reasoning": "preserve task",
                        "answer": f"answer {r['id']}",
                    }
                    for r in rows
                ]
            )
        )


class Tests(unittest.TestCase):
    def fixture(self, root):
        bp = root / "baseline.jsonl"
        bp.write_text(
            json.dumps(
                {
                    "skill": "SR_INSTRUCTION_FOLLOWING",
                    "difficulty": "clean_fit",
                    "problem": "baseline prompt",
                    "reasoning": "r",
                    "answer": "a",
                }
            )
            + "\n"
            + json.dumps(
                {
                    "skill": "DED",
                    "difficulty": "L1",
                    "problem": "logic anchor",
                    "reasoning": "r",
                    "answer": "a",
                }
            )
            + "\n"
        )
        bm = root / "baseline.manifest.json"
        bm.write_text(
            json.dumps(
                {
                    "schema": "small-llm-superior-reasoning-production-v1",
                    "policy": base.superior.PRODUCTION_FILTER_VERSION,
                    "context_length": 2048,
                    "source_rows": 4,
                    "domain_counts": {"instruction_following": 4},
                    "valid_unique_instruction_rows": 4,
                    "rejected_output_count": 0,
                    "duplicate_input_count": 0,
                    "exclusion_counts": {"over_context": 3},
                    "selected_count": 1,
                    "gemini_rows": 1,
                    "combined_rows": 2,
                    "output_sha256": sha(bp),
                }
            )
        )
        w = root / "work"
        base.prepare_candidates(
            w,
            baseline_manifest=bm,
            rows=[
                src(0, "baseline prompt", "short"),
                src(1, "long keep one"),
                src(2, "long excluded"),
                src(3, "long keep two"),
            ],
            token_counter=lambda t: len(t.split()),
            progress_every=0,
        )
        c = w / "curation.jsonl"
        c.write_text(
            "".join(
                json.dumps(
                    {
                        "schema": base.MANUAL_CURATION_SCHEMA,
                        "id": rid,
                        "decision": d,
                        "reason": "test",
                    }
                )
                + "\n"
                for rid, d in [
                    ("source-1", "keep"),
                    ("source-2", "exclude_code"),
                    ("source-3", "keep"),
                ]
            )
        )
        return bp, bm, w, c

    def test_keeper_only_resume_and_finalize(self):
        with tempfile.TemporaryDirectory() as d:
            bp, bm, w, c = self.fixture(Path(d))
            old_factory = base.transport.GeminiDistillationClient
            base.transport.GeminiDistillationClient = Client
            try:
                old = base.adapt_candidates(
                    w,
                    batch_size=1,
                    max_attempts=1,
                    retry_delay_seconds=0,
                    request_interval_seconds=0,
                    max_batches=1,
                    client=Client(),
                )
                self.assertFalse(old["complete"])
                p = resume.prepare_keep_resume(
                    w, manual_curation_jsonl=c, baseline_manifest=bm, batch_size=1
                )
                self.assertEqual(p["original_accepted_kept_records"], 1)
                self.assertEqual(p["pending_kept_records"], 1)
                self.assertEqual(
                    [r["id"] for r in base._read_jsonl(Path(p["candidates_jsonl"]))],
                    ["source-3"],
                )
                wave = resume.adapt_keep_wave(
                    w,
                    first_batch=1,
                    batch_count=1,
                    workers=1,
                    max_attempts=1,
                    retry_delay_seconds=0,
                )
                self.assertEqual(wave["completed_records"], 1)
                st = resume.keep_status(w)
                self.assertEqual(st["resume_pending_records"], 0)
                self.assertEqual(st["total_accepted_kept_records"], 2)
                out = Path(d) / "complete.jsonl"
                f = resume.finalize_curated_complete_dataset(
                    w,
                    baseline_jsonl=bp,
                    baseline_manifest=bm,
                    manual_curation_jsonl=c,
                    output_jsonl=out,
                )
                self.assertEqual(f["accepted_keep_rewrites"], 2)
                self.assertEqual(f["duplicate_rewrite_exclusions"], 0)
                self.assertEqual(f["combined_rows"], 4)
            finally:
                base.transport.GeminiDistillationClient = old_factory

    def test_prepare_resume_identity(self):
        with tempfile.TemporaryDirectory() as d:
            _, bm, w, c = self.fixture(Path(d))
            a = resume.prepare_keep_resume(
                w, manual_curation_jsonl=c, baseline_manifest=bm, batch_size=1
            )
            b = resume.prepare_keep_resume(
                w, manual_curation_jsonl=c, baseline_manifest=bm, batch_size=1
            )
            self.assertFalse(a["resumed_complete"])
            self.assertTrue(b["resumed_complete"])
            self.assertEqual(b["pending_kept_records"], 2)


if __name__ == "__main__":
    unittest.main()
