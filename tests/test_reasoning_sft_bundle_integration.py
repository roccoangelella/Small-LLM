from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from trainer.identity import canonical_hash
from post_training.sft.bundle import BUNDLE_SCHEMA_VERSION, verify_bundle
from post_training.sft.config import SFTDataConfig
from post_training.sft.mixture import build_atomic_blocks
from post_training.sft.schema import TokenizedSFTRecord
from post_training.sft.storage import SFTDatasetWriter


REPO = Path(__file__).resolve().parents[1]
RSFT_DIR = REPO / "post_training" / "R-SFT"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load(name: str):
    module_name = f"small_llm_rsft_bundle_integration_{name}"
    path = RSFT_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


bundle = _load("bundle")

SOURCES = {
    "smol-magpie-ultra-short": 0.75,
    "smol-contraints": 0.10,
    "smollm-rewrite-30k": 0.075,
    "smol-summarize-20k": 0.075,
}


def _record(record_id: str, source: str, split: str, value: int) -> TokenizedSFTRecord:
    return TokenizedSFTRecord(
        record_id=record_id,
        source=source,
        split=split,  # type: ignore[arg-type]
        token_ids=(50_256, value % 50_000, (value + 1) % 50_000),
        target_mask=(True, True),
    )


def _write_split(root: Path, split: str, records: list[TokenizedSFTRecord]) -> dict[str, object]:
    total = sum(record.target_token_count for record in records)
    config = SFTDataConfig(
        target_loss_tokens=total,
        optimizer_target_tokens=32,
        context_length=2_048,
        maximum_assistant_tokens=512,
        instruction_share=1.0,
        replay_share=0.0,
        instruction_source_shares=SOURCES,
        seed=17,
    )
    manifest = SFTDatasetWriter(root, config).write(
        build_atomic_blocks(records, target_tokens_per_block=32)
    )
    report_without_hash: dict[str, object] = {
        "schema": "small-llm-sft-build-report",
        "manifest_identity": manifest["manifest_sha256"],
        "source_audit": {},
        "actual_source_target_tokens": manifest["totals"]["source_target_tokens"],  # type: ignore[index]
    }
    report = {
        **report_without_hash,
        "report_sha256": canonical_hash(report_without_hash),
    }
    (root / "build-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "path": split,
        "manifest_sha256": manifest["manifest_sha256"],
        "loss_bearing_target_tokens": manifest["totals"]["loss_bearing_target_tokens"],  # type: ignore[index]
        "build_report_sha256": report["report_sha256"],
    }


def _make_s0_bundle(root: Path) -> Path:
    root.mkdir()
    train: list[TokenizedSFTRecord] = []
    value = 100
    for index in range(40):
        for source in SOURCES:
            train.append(_record(f"train-{source}-{index}", source, "train", value))
            value += 3
    validation = [
        _record(f"validation-{source}", source, "validation", value + index * 3)
        for index, source in enumerate(SOURCES)
    ]
    test = [
        _record(f"test-{source}", source, "test", value + 100 + index * 3)
        for index, source in enumerate(SOURCES)
    ]
    splits = {
        "train": _write_split(root / "train", "train", train),
        "validation": _write_split(root / "validation", "validation", validation),
        "test": _write_split(root / "test", "test", test),
    }
    source_without_hash = {
        "schema": "synthetic-s0-source-v1",
        "name": "unit-test-s0",
    }
    source = {
        **source_without_hash,
        "manifest_sha256": canonical_hash(source_without_hash),
    }
    (root / "source-manifest.json").write_text(
        json.dumps(source, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    train_targets = int(splits["train"]["loss_bearing_target_tokens"])
    manifest_without_hash: dict[str, object] = {
        "schema": "small-llm-sft-bundle",
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "prepared_source_manifest_sha256": source["manifest_sha256"],
        "prepared_source": {
            "dataset_name": "synthetic-s0",
            "revision": "unit-test",
            "license_id": "test",
            "split_policy": {"kind": "unit-test"},
        },
        "train_target_tokens_requested": train_targets,
        "optimizer_target_tokens": 32,
        "context_length": 2_048,
        "maximum_assistant_tokens": 512,
        "instruction_share": 0.85,
        "replay_share": 0.15,
        "instruction_source_shares": SOURCES,
        "seed": 17,
        "splits": splits,
    }
    manifest = {
        **manifest_without_hash,
        "manifest_sha256": canonical_hash(manifest_without_hash),
    }
    (root / "bundle-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    verify_bundle(root)
    return root


def _make_reasoning_jsonl(path: Path) -> None:
    records = []
    for skill in bundle.prompts.R0_SKILLS:
        for difficulty in bundle.generation.R0_DIFFICULTIES:
            for index in range(3):
                records.append(
                    bundle.schema.ReasoningExample(
                        skill=skill,
                        difficulty=difficulty,
                        problem=f"{skill} {difficulty} problem {index}: what follows?",
                        reasoning=f"Use the supplied facts to infer result {index}.",
                        answer=f"Result {index} follows.",
                    )
                )
    bundle.schema.write_jsonl(records, path)


class ReasoningSFTBundleIntegrationTests(unittest.TestCase):
    def test_matched_bundles_are_native_sft_bundles_with_identical_retention(self) -> None:
        try:
            import tiktoken  # noqa: F401
        except ImportError:
            self.skipTest("tiktoken is required for R-SFT bundle tokenization")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            s0 = _make_s0_bundle(root / "s0")
            reasoning = root / "reasoning.jsonl"
            _make_reasoning_jsonl(reasoning)
            token_spec = root / "reasoning-tokens.json"
            token_spec.write_text(
                json.dumps(
                    {
                        "reasoning_start": "<|rsft_reasoning|>",
                        "reasoning_end": "<|rsft_end_reasoning|>",
                        "answer_start": "<|rsft_answer|>",
                    }
                ),
                encoding="utf-8",
            )
            output = root / "matched"
            pilot = bundle.build_matched_pilot_bundles(
                reasoning,
                s0_bundle=s0,
                token_spec_path=token_spec,
                output_dir=output,
                examples_per_cell=3,
                heldout_per_cell=1,
                optimizer_target_tokens=64,
                seed=17,
            )

            atomic = verify_bundle(output / "atomic")
            textual = verify_bundle(output / "textual")
            self.assertEqual(atomic["status"], "verified")
            self.assertEqual(textual["status"], "verified")
            self.assertTrue((output / "atomic" / "reasoning-tokens.json").is_file())
            self.assertTrue((output / "textual" / "reasoning-tokens.json").is_file())
            self.assertEqual(pilot["examples"]["total"], 63)
            self.assertEqual(pilot["examples"]["train"], 21)
            self.assertEqual(pilot["examples"]["validation"], 21)
            self.assertEqual(pilot["examples"]["test"], 21)

            atomic_sources = pilot["arms"]["atomic"]["train_source_target_tokens"]
            textual_sources = pilot["arms"]["textual"]["train_source_target_tokens"]
            for source in SOURCES:
                self.assertEqual(atomic_sources[source], textual_sources[source])
            self.assertNotIn("climbmix-replay", atomic_sources)
            self.assertNotIn("climbmix-replay", textual_sources)


if __name__ == "__main__":
    unittest.main()
