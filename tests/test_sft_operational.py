from __future__ import annotations

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
from post_training.sft.schema import ChatMessage, ConversationRecord, SFTBlock, TokenizedSFTRecord
from trainer.step import _microbatch_to_device, _ordered_batch_tensors


class SFTOperationalTests(unittest.TestCase):
    def test_500m_budget_uses_exact_parent_counter(self) -> None:
        self.assertEqual(sft_budget_from_parent(500_156_416), 20_006_256)

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


if __name__ == "__main__":
    unittest.main()
