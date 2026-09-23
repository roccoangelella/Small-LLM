"""Regression contracts for document-boundary SuperBPE corpus production."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from dataset import config
from dataset.moe_100b import (
    RUN_ID,
    TARGET_SOURCE_TOKENS,
    production_arguments,
)
from dataset.production.policy import ProductionPolicy, configuration_hash, schema_hash
from dataset.src.records import ParsedRecord
from dataset.src.streaming import SequencePacker, SourceDocument, StreamCacheConfig, synthetic_test_weights
from dataset.src.verify import _scan_uint16_ranges
from dataset.src.workplan import WorkPlan
from dataset.superbpe_retokenization import (
    DEFAULT_TOKENIZER_CONTRACT_ID,
    EXPECTED_SPECIAL_TOKENS,
    SOURCE_EOD_TOKEN_ID,
    SUPERBPE_8000,
    SUPERBPE_8000_V2,
    TARGET_EOD_TOKEN_ID,
    TARGET_SEMANTIC_VOCAB_SIZE,
    TARGET_SOURCE_VOCAB_SIZE,
    TOKENIZER_IDENTITIES,
    install_superbpe_retokenization,
    resolve_tokenizer_identity,
)

ROOT = Path(__file__).resolve().parents[1]


def _stream() -> StreamCacheConfig:
    return StreamCacheConfig(
        context_length=8,
        sequences_per_block=2,
        target_shard_bytes=1024,
        reader_workers=1,
        max_in_flight_work_items=1,
        per_cluster_queue_limit=4,
        prepared_block_queue_limit=4,
        prefetch_head_start=0,
        weights=synthetic_test_weights(),
        scheduler_tie_break_seed="test",
    )


def _plan() -> WorkPlan:
    return WorkPlan(
        schema_version=config.WORK_PLAN_SCHEMA_VERSION,
        dataset="repo",
        revision="rev",
        source_glob="part_*.jsonl",
        selection_seed="seed",
        region_bytes=1,
        source_files=(),
        work_items=(),
        hash="a" * 64,
    )


class SuperBPERetokenizationTests(unittest.TestCase):
    def _source_record(self, text: str) -> ParsedRecord:
        import tiktoken

        ids = list(tiktoken.get_encoding("gpt2").encode(text)) + [SOURCE_EOD_TOKEN_ID]
        return ParsedRecord(
            17,
            json.dumps(
                {"cluster_id": 1, "tokens": ids, "token_count": len(ids)}
            ).encode("utf-8"),
        )

    def test_document_is_retokenized_before_source_token_accounting(self) -> None:
        from dataset.src import streaming
        from tokenizers import Tokenizer

        original_eod = config.EOD_TOKEN_ID
        original_validator = streaming.validate_record
        installed = install_superbpe_retokenization(ROOT)
        try:
            text = "The quick brown fox jumps over the lazy dog."
            validated = streaming.validate_record(self._source_record(text))
            self.assertTrue(validated.valid)
            assert validated.tokens is not None
            self.assertTrue(validated.tokens)
            self.assertTrue(all(0 <= token < TARGET_SOURCE_VOCAB_SIZE for token in validated.tokens))
            self.assertNotIn(SOURCE_EOD_TOKEN_ID, validated.tokens)

            tokenizer = Tokenizer.from_file(str(ROOT / "tokenizer" / "superbpe_8000.json"))
            self.assertEqual(tokenizer.decode(list(validated.tokens), skip_special_tokens=False), text)

            document = SourceDocument("doc", 1, validated.tokens)
            self.assertEqual(document.source_token_count, len(validated.tokens))
        finally:
            installed.restore()
        self.assertEqual(config.EOD_TOKEN_ID, original_eod)
        self.assertIs(streaming.validate_record, original_validator)

    def test_reserved_control_strings_remain_ordinary_source_text(self) -> None:
        from dataset.src import streaming

        installed = install_superbpe_retokenization(ROOT)
        try:
            validated = streaming.validate_record(
                self._source_record("literal <think> tag in a web page")
            )
            self.assertTrue(validated.valid)
            assert validated.tokens is not None
            self.assertNotIn(7_993, validated.tokens)
            self.assertTrue(all(token < TARGET_SOURCE_VOCAB_SIZE for token in validated.tokens))
        finally:
            installed.restore()

    def test_existing_packer_inserts_superbpe_eod_only(self) -> None:
        installed = install_superbpe_retokenization(ROOT)
        try:
            packer = SequencePacker(8)
            packer.push(SourceDocument("doc", 1, (1, 2, 3)))
            carry = packer.state_dict()["carry_tokens"]
            self.assertEqual(carry[-1], TARGET_EOD_TOKEN_ID)
            self.assertNotIn(SOURCE_EOD_TOKEN_ID, carry)
        finally:
            installed.restore()

    def test_superbpe_identity_changes_hashes_but_restore_recovers_legacy(self) -> None:
        stream = _stream()
        plan = _plan()
        policy = ProductionPolicy("unit-test", target_source_tokens=10,
                                  minimum_source_tokens=9, maximum_source_tokens=11,
                                  checkpoint_source_tokens=2, remote_required=False)
        legacy_config = configuration_hash(policy, stream, plan)
        legacy_schema = schema_hash(stream)

        installed = install_superbpe_retokenization(ROOT)
        try:
            self.assertNotEqual(configuration_hash(policy, stream, plan), legacy_config)
            self.assertNotEqual(schema_hash(stream), legacy_schema)
            self.assertEqual(installed.contract["semantic_vocab_size"], 8_000)
            self.assertEqual(installed.contract["eod_token_id"], 7_992)
        finally:
            installed.restore()
        self.assertEqual(configuration_hash(policy, stream, plan), legacy_config)
        self.assertEqual(schema_hash(stream), legacy_schema)

    def test_strict_uint16_scan_uses_8000_semantic_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "shard.bin"
            path.write_bytes((7_999).to_bytes(2, "little") + (8_000).to_bytes(2, "little"))
            problem = _scan_uint16_ranges(
                path,
                semantic_vocab_size=TARGET_SEMANTIC_VOCAB_SIZE,
            )
            self.assertIsNotNone(problem)
            self.assertIn("8000", str(problem))


class MoE100BProfileTests(unittest.TestCase):
    def test_frozen_profile_reuses_existing_incremental_hf_pipeline(self) -> None:
        args = production_arguments(
            ["--weights-file", "dataset/climbmix_code_free_weights.json", "--output-dir", "out"]
        )
        self.assertEqual(args[args.index("--run-id") + 1], RUN_ID)
        self.assertEqual(args[args.index("--tokenizer-contract") + 1], "superbpe_8000")
        self.assertEqual(args[args.index("--target-tokens") + 1], str(TARGET_SOURCE_TOKENS))
        self.assertEqual(args[args.index("--sequences-per-block") + 1], "64")
        self.assertEqual(args[args.index("--target-shard-bytes") + 1], str(1024**3))
        self.assertIn("--public-hf-bucket", args)
        self.assertIn("--evict-remote-shards", args)
        self.assertIn("--incremental-frontier", args)
        self.assertEqual(args[args.index("--nominal-training-tokens") + 1], str(100_000_000_000))

    def test_scientific_identity_cannot_be_overridden_from_cli(self) -> None:
        for flag in (
            "--tokenizer-contract",
            "--target-tokens",
            "--run-id",
            "--context-length",
            "--public-hf-bucket",
        ):
            with self.subTest(flag=flag), self.assertRaises(SystemExit):
                supplied = ["--weights-file", "weights.json", "--output-dir", "out", flag]
                if flag != "--public-hf-bucket":
                    supplied.append("bad")
                production_arguments(supplied)

    def test_modal_producer_targets_and_verifies_a_public_bucket(self) -> None:
        source = (ROOT / "modal" / "moe_100b_dataset.py").read_text(encoding="utf-8")
        self.assertIn("dataset_bucket_id: str", source)
        self.assertIn('_REMOTE_REPO_ROOT = Path("/root/small-llm")', source)
        self.assertIn("_base._with_local_repo(", source)
        self.assertIn("_base.IMAGE_BASE.uv_pip_install(", source)
        self.assertIn('"--hf-bucket-id", bucket_id', source)
        self.assertIn("private=False", source)
        self.assertIn("store.verify_bucket_visibility()", source)


def _git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1()  # noqa: S324 - Git object identity, not security
    digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _source_text_tokenizer(path: Path):
    """Load an 8k artifact the way the producer does: specials unreachable from text."""
    from tokenizers import Tokenizer

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["added_tokens"] = []
    payload["model"]["vocab"] = {
        token: token_id
        for token, token_id in payload["model"]["vocab"].items()
        if token_id < TARGET_SOURCE_VOCAB_SIZE
    }
    return Tokenizer.from_str(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


class TokenizerContractSelectionTests(unittest.TestCase):
    """Two frozen 8k artifacts share geometry but not vocabulary (ADR 0184)."""

    def test_default_contract_is_still_run_001_artifact(self) -> None:
        self.assertEqual(DEFAULT_TOKENIZER_CONTRACT_ID, "superbpe_8000")
        self.assertIs(resolve_tokenizer_identity(), SUPERBPE_8000)
        self.assertEqual(SUPERBPE_8000.relative_path, "tokenizer/superbpe_8000.json")
        self.assertEqual(
            SUPERBPE_8000.expected_git_blob_sha1,
            "a4daaa638a4d9db10270a65e8a6b55a9b94a9fd4",
        )

    def test_every_registered_artifact_matches_its_pin_on_disk(self) -> None:
        for contract_id, identity in sorted(TOKENIZER_IDENTITIES.items()):
            with self.subTest(contract=contract_id):
                path = ROOT / identity.relative_path
                self.assertTrue(path.is_file(), f"missing tokenizer artifact {path}")
                self.assertEqual(_git_blob_sha1(path), identity.expected_git_blob_sha1)

    def test_v2_artifact_is_the_corpus_v2_file(self) -> None:
        path = ROOT / SUPERBPE_8000_V2.relative_path
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "068e20ef8a16b1bff2a2a1d161e5a6f1cfd0935befa7c0ae734c9ec376d373bb",
        )
        self.assertEqual(
            SUPERBPE_8000_V2.expected_git_blob_sha1,
            "961597b0a1196a5dbfe1c0a18f2659f908b425ae",
        )

    def test_both_artifacts_share_geometry_but_not_vocabulary(self) -> None:
        payloads = {
            contract_id: json.loads(
                (ROOT / identity.relative_path).read_text(encoding="utf-8")
            )
            for contract_id, identity in TOKENIZER_IDENTITIES.items()
        }
        for contract_id, payload in payloads.items():
            with self.subTest(contract=contract_id):
                vocab = payload["model"]["vocab"]
                self.assertEqual(
                    sorted(vocab.values()), list(range(TARGET_SEMANTIC_VOCAB_SIZE))
                )
                self.assertEqual(
                    {i: t for t, i in vocab.items() if i >= TARGET_SOURCE_VOCAB_SIZE},
                    EXPECTED_SPECIAL_TOKENS,
                )
        legacy, v2 = payloads["superbpe_8000"], payloads["superbpe_8000_v2"]
        self.assertEqual(legacy["pre_tokenizer"], v2["pre_tokenizer"])
        self.assertEqual(legacy["decoder"], v2["decoder"])
        self.assertEqual(legacy["added_tokens"], v2["added_tokens"])
        self.assertNotEqual(legacy["model"]["vocab"], v2["model"]["vocab"])
        self.assertNotEqual(legacy["model"]["merges"], v2["model"]["merges"])

    def test_the_two_artifacts_encode_ordinary_text_differently(self) -> None:
        legacy = _source_text_tokenizer(ROOT / SUPERBPE_8000.relative_path)
        v2 = _source_text_tokenizer(ROOT / SUPERBPE_8000_V2.relative_path)
        for text in (
            "The quick brown fox jumps over the lazy dog.",
            "Mixture-of-experts routing keeps the active parameter count small.",
            "A corpus must never mix two tokenizers that share a vocabulary size.",
        ):
            with self.subTest(text=text):
                legacy_ids = legacy.encode(text, add_special_tokens=False).ids
                v2_ids = v2.encode(text, add_special_tokens=False).ids
                self.assertNotEqual(legacy_ids, v2_ids)
                self.assertEqual(legacy.decode(legacy_ids, skip_special_tokens=False), text)
                self.assertEqual(v2.decode(v2_ids, skip_special_tokens=False), text)

    def test_installing_v2_reports_the_v2_artifact_and_restores(self) -> None:
        from dataset.src import streaming

        original_eod = config.EOD_TOKEN_ID
        original_validator = streaming.validate_record
        installed = install_superbpe_retokenization(
            ROOT, tokenizer_contract="superbpe_8000_v2"
        )
        try:
            contract = installed.contract
            self.assertEqual(contract["output_tokenizer_id"], "superbpe_8000_v2")
            self.assertEqual(contract["tokenizer_artifact"], "tokenizer/superbpe_8000_v2.json")
            self.assertEqual(
                contract["tokenizer_git_blob_sha1"], SUPERBPE_8000_V2.expected_git_blob_sha1
            )
            self.assertEqual(
                contract["tokenizer_sha256"],
                "068e20ef8a16b1bff2a2a1d161e5a6f1cfd0935befa7c0ae734c9ec376d373bb",
            )
            self.assertEqual(contract["semantic_vocab_size"], TARGET_SEMANTIC_VOCAB_SIZE)
            self.assertEqual(contract["eod_token_id"], TARGET_EOD_TOKEN_ID)
            self.assertEqual(config.CORPUS_OUTPUT_TOKENIZER_ID, "superbpe_8000_v2")
        finally:
            installed.restore()
        self.assertEqual(config.EOD_TOKEN_ID, original_eod)
        self.assertIs(streaming.validate_record, original_validator)
        self.assertIsNone(getattr(config, "CORPUS_OUTPUT_TOKENIZER_ID", None))

    def test_default_installation_still_reports_run_001_artifact(self) -> None:
        installed = install_superbpe_retokenization(ROOT)
        try:
            contract = installed.contract
            self.assertEqual(contract["output_tokenizer_id"], "superbpe_8000")
            self.assertEqual(contract["tokenizer_artifact"], "tokenizer/superbpe_8000.json")
            self.assertEqual(
                contract["tokenizer_git_blob_sha1"], SUPERBPE_8000.expected_git_blob_sha1
            )
        finally:
            installed.restore()

    def test_unknown_contract_id_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_tokenizer_identity("superbpe_8000_v3")
        with self.assertRaises(ValueError):
            install_superbpe_retokenization(ROOT, tokenizer_contract="gpt2")

    def test_production_cli_exposes_both_superbpe_contracts(self) -> None:
        from dataset.production.cli import build_parser

        action = next(
            item
            for item in build_parser()._actions
            if item.dest == "tokenizer_contract"
        )
        self.assertEqual(action.default, "gpt2")
        self.assertEqual(
            sorted(action.choices), ["gpt2", "superbpe_8000", "superbpe_8000_v2"]
        )

    def test_only_the_dataset_producer_loads_a_tokenizer_artifact(self) -> None:
        """The trainer/inference side reads semantic_vocab_size, never the 8k JSON."""
        offenders = []
        for path in sorted(ROOT.rglob("*.py")):
            relative = path.relative_to(ROOT)
            head = relative.parts[0]
            if head in {"dataset", "tests", ".venv", "build", "__pycache__"}:
                continue
            if "superbpe_8000" in path.read_text(encoding="utf-8", errors="ignore"):
                offenders.append(str(relative))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
