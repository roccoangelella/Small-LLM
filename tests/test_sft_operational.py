from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch

from post_training.sft.behavior_eval import BehaviorCase, verify_response
from post_training.sft.bundle import (
    IdentitySplitPolicy,
    conversation_content_hash,
    conversation_group_id,
    sft_budget_from_parent,
)
from post_training.sft.checkpoints import sft_checkpoint_hashes
from post_training.sft.config import SFTDataConfig
from post_training.sft.eval_suite import _deltas
from post_training.sft.schema import ChatMessage, ConversationRecord, SFTBlock, TokenizedSFTRecord
from post_training.sft.train_cli import _checkpoint_step
from trainer.evaluation import evaluate_batches
from trainer.step import _microbatch_to_device, _ordered_batch_tensors
from trainer.types import TokenBatch
from tests.trainer_fixtures import TinyLM


def _load_sft_runtime():
    runtime_path = Path(__file__).resolve().parents[1] / "kaggle" / "sft_runtime.py"
    spec = importlib.util.spec_from_file_location("small_llm_sft_runtime_test", runtime_path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"cannot load SFT runtime from {runtime_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sft_runtime = _load_sft_runtime()


class SFTOperationalTests(unittest.TestCase):
    def test_500m_budget_uses_exact_parent_counter(self) -> None:
        self.assertEqual(sft_budget_from_parent(500_156_416), 20_006_256)

    def test_portable_work_root_defaults_beside_repo_off_kaggle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "Small-LLM"
            repo.mkdir()
            resolved = sft_runtime._portable_work_root(
                None,
                kaggle_work=root / "missing-kaggle-working",
                repo=repo,
            )
            self.assertEqual(resolved, (root / "small-llm-work").resolve())

    def test_portable_work_root_honors_explicit_override(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configured = root / "custom-work"
            resolved = sft_runtime._portable_work_root(
                str(configured),
                kaggle_work=root / "missing-kaggle-working",
                repo=root / "Small-LLM",
            )
            self.assertEqual(resolved, configured.resolve())

    def test_replay_root_requires_dataset_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "dataset"
            dataset.mkdir()
            manifest = dataset / "manifest.json"
            manifest.write_text("{}\n", encoding="utf-8")
            self.assertEqual(sft_runtime._resolve_replay_root(str(dataset)), dataset.resolve())
            with self.assertRaisesRegex(sft_runtime.RuntimeFailure, "must be the pretraining dataset directory"):
                sft_runtime._resolve_replay_root(str(manifest))

    def test_prompt_derivatives_share_split_group(self) -> None:
        first = ConversationRecord(
            "a",
            "smol-magpie-ultra-short",
            (
                ChatMessage("user", "What is two plus two?"),
                ChatMessage("assistant", "4"),
            ),
        )
        second = ConversationRecord(
            "b",
            "smol-magpie-ultra-short",
            (
                ChatMessage("user", "What is two plus two?"),
                ChatMessage("assistant", "Four."),
            ),
        )
        self.assertEqual(conversation_group_id(first), conversation_group_id(second))
        self.assertNotEqual(conversation_content_hash(first), conversation_content_hash(second))
        policy = IdentitySplitPolicy()
        self.assertEqual(
            policy.assign(conversation_group_id(first)),
            policy.assign(conversation_group_id(second)),
        )

    def test_validation_mixture_can_be_instruction_only(self) -> None:
        config = SFTDataConfig(
            instruction_share=1.0,
            replay_share=0.0,
            instruction_source_shares={"a": 1.0},
        )
        self.assertEqual(config.complete_source_shares, {"a": 1.0})

    def test_masked_rows_are_length_bucketed_and_cropped_per_microbatch(self) -> None:
        long_record = TokenizedSFTRecord(
            "long",
            "a",
            "train",
            (1, 2, 3, 4, 5, 6),
            (False, False, True, True, True),
        )
        short_record = TokenizedSFTRecord(
            "short",
            "a",
            "train",
            (7, 8, 9),
            (False, True),
        )
        batch = SFTBlock(0, "train", (long_record, short_record)).to_token_batch(
            pad_token_id=50_256
        )
        inputs, labels = _ordered_batch_tensors(batch)
        self.assertEqual(inputs[0, 0].item(), 7)
        cropped_inputs, cropped_labels = _microbatch_to_device(
            inputs,
            labels,
            start=0,
            stop=1,
            device=torch.device("cpu"),
        )
        self.assertEqual(cropped_inputs.shape, (1, 2))
        self.assertEqual(cropped_labels.shape, (1, 2))
        self.assertEqual(batch.target_token_count, 4)

    def test_behavior_verifier_requires_eos_and_blocks_role_leak(self) -> None:
        case = BehaviorCase(
            "exact",
            "constraints",
            (ChatMessage("user", "reply yes"),),
            exact="yes",
            maximum_words=1,
        )
        self.assertTrue(
            verify_response(
                case,
                text="yes",
                response_token_ids=(1,),
                terminated_with_eos=True,
            )["passed"]
        )
        leaked = verify_response(
            case,
            text="yes\nUser: next",
            response_token_ids=(1, 2),
            terminated_with_eos=True,
        )
        self.assertFalse(leaked["passed"])
        self.assertTrue(leaked["role_leak"])

    def test_checkpoint_identity_binds_parent(self) -> None:
        bundle = {
            "manifest_sha256": "b" * 64,
            "context_length": 2048,
            "optimizer_target_tokens": 32768,
            "splits": {"train": {"manifest_sha256": "c" * 64}},
        }
        trainer = {"learning_rate": 3e-5, "microbatch_size": 4}
        first = sft_checkpoint_hashes(
            parent_identity={"identity_sha256": "a" * 64},
            bundle_manifest=bundle,
            trainer_config=trainer,
        )
        second = sft_checkpoint_hashes(
            parent_identity={"identity_sha256": "d" * 64},
            bundle_manifest=bundle,
            trainer_config=trainer,
        )
        self.assertNotEqual(first[0], second[0])

    def test_checkpoint_step_is_strict(self) -> None:
        self.assertEqual(_checkpoint_step("step-00000250"), 250)
        with self.assertRaises(RuntimeError):
            _checkpoint_step("step-250")

    def test_held_out_evaluator_accepts_test_split(self) -> None:
        model = TinyLM()
        engine = SimpleNamespace(
            model=model,
            device=torch.device("cpu"),
            config=SimpleNamespace(precision="fp32"),
            optimizer=None,
            best_validation_loss=None,
        )
        batch = TokenBatch(
            block_id=0,
            split="test",
            input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
            labels=torch.tensor([[-100, 3, 4]], dtype=torch.long),
            sequence_count=1,
            target_token_count=2,
        )
        result = evaluate_batches(engine, [batch], maximum_batches=1)
        self.assertEqual(result["target_tokens"], 2)
        self.assertIsNone(engine.best_validation_loss)

    def test_comprehensive_deltas_include_nested_capabilities(self) -> None:
        parent = {
            "eval_core_v1": {
                "loss": 4.0,
                "perplexity": 54.0,
                "bits_per_byte": 1.5,
                "cluster_macro_loss": 4.1,
                "cluster_mixture_weighted_loss": 4.0,
                "worst_cluster_loss": 4.5,
                "tokens_per_second": 100.0,
                "peak_allocated_vram_bytes": 10,
                "top_k_accuracy": {"1": 0.2, "5": 0.4},
                "calibration": {"ece": 0.10},
                "per_cluster": {"0": {"loss": 4.0, "perplexity": 54.0}},
                "position_buckets": [{"index": 0, "loss": 4.0}],
            },
            "sft_validation": {"loss": 3.0, "perplexity": 20.0},
            "sft_test": {"loss": 3.1, "perplexity": 22.0},
            "instruction_behavior": {
                "summary": {
                    "pass_rate": 0.1,
                    "eos_termination_rate": 0.2,
                    "runaway_rate": 0.8,
                    "empty_rate": 0.1,
                    "role_leak_rate": 0.1,
                    "mean_response_tokens": 20.0,
                    "mean_trigram_repetition": 0.2,
                },
                "per_category": {"direct_qa": {"passed": 1, "pass_rate": 0.2}},
            },
        }
        tuned = {
            "eval_core_v1": {
                "loss": 4.2,
                "perplexity": 60.0,
                "bits_per_byte": 1.6,
                "cluster_macro_loss": 4.3,
                "cluster_mixture_weighted_loss": 4.2,
                "worst_cluster_loss": 4.7,
                "tokens_per_second": 95.0,
                "peak_allocated_vram_bytes": 12,
                "top_k_accuracy": {"1": 0.3, "5": 0.5},
                "calibration": {"ece": 0.08},
                "per_cluster": {"0": {"loss": 4.2, "perplexity": 60.0}},
                "position_buckets": [{"index": 0, "loss": 4.2}],
            },
            "sft_validation": {"loss": 2.0, "perplexity": 10.0},
            "sft_test": {"loss": 2.1, "perplexity": 11.0},
            "instruction_behavior": {
                "summary": {
                    "pass_rate": 0.6,
                    "eos_termination_rate": 0.9,
                    "runaway_rate": 0.1,
                    "empty_rate": 0.0,
                    "role_leak_rate": 0.0,
                    "mean_response_tokens": 12.0,
                    "mean_trigram_repetition": 0.05,
                },
                "per_category": {"direct_qa": {"passed": 3, "pass_rate": 0.6}},
            },
        }
        deltas = _deltas(parent, tuned)
        self.assertAlmostEqual(deltas["eval_core_v1"]["top_k_accuracy"]["1"], 0.1)
        self.assertAlmostEqual(deltas["eval_core_v1"]["calibration_ece"], -0.02)
        self.assertAlmostEqual(
            deltas["instruction_behavior"]["per_category"]["direct_qa"]["pass_rate"],
            0.4,
        )
        self.assertAlmostEqual(deltas["sft_test"]["loss"], -1.0)


if __name__ == "__main__":
    unittest.main()
