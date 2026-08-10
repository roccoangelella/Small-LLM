from __future__ import annotations

import unittest

from post_training.sft.bundle import conversation_content_hash, conversation_group_id
from post_training.sft.schema import ChatMessage, ConversationRecord


class SFTGlobalIdentityTests(unittest.TestCase):
    def test_same_prompt_aliases_group_across_source_labels(self) -> None:
        first = ConversationRecord(
            "a",
            "smol-magpie-ultra-short",
            (
                ChatMessage("user", "  What is Two PLUS two?  "),
                ChatMessage("assistant", "4"),
            ),
        )
        second = ConversationRecord(
            "b",
            "smol-contraints",
            (
                ChatMessage("user", "what is two plus TWO?"),
                ChatMessage("assistant", "Four."),
            ),
        )
        self.assertEqual(conversation_group_id(first), conversation_group_id(second))
        self.assertNotEqual(conversation_content_hash(first), conversation_content_hash(second))

    def test_exact_duplicate_hash_ignores_source_label(self) -> None:
        messages = (
            ChatMessage("user", "Return the third item."),
            ChatMessage("assistant", "mango"),
        )
        first = ConversationRecord("a", "smol-magpie-ultra-short", messages)
        second = ConversationRecord("b", "smol-contraints", messages)
        self.assertEqual(conversation_content_hash(first), conversation_content_hash(second))


if __name__ == "__main__":
    unittest.main()
