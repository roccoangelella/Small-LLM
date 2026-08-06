from __future__ import annotations

import unittest

from post_training.sft.schema import ChatMessage, ConversationRecord
from post_training.sft.template import GPT2ChatTemplate


class CharacterEncoder:
    def encode(self, text: str) -> list[int]:
        return [ord(character) + 1 for character in text]


class TemplateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.encoder = CharacterEncoder()
        self.template = GPT2ChatTemplate(
            eos_token_id=60000,
            maximum_context_tokens=4096,
            maximum_assistant_tokens=512,
        )

    def test_only_assistant_content_and_turn_eos_bear_loss(self) -> None:
        record = ConversationRecord(
            "example",
            "smol-magpie-ultra-short",
            (
                ChatMessage("user", "Question"),
                ChatMessage("assistant", "Answer"),
            ),
        )
        tokenized = self.template.encode_conversation(record, self.encoder)
        active = [
            tokenized.token_ids[index + 1]
            for index, enabled in enumerate(tokenized.target_mask)
            if enabled
        ]
        self.assertEqual(active, self.encoder.encode("Answer") + [60000])
        self.assertEqual(tokenized.target_token_count, len("Answer") + 1)

    def test_every_assistant_turn_has_its_own_stop_target(self) -> None:
        record = ConversationRecord(
            "multi",
            "smol-magpie-ultra-short",
            (
                ChatMessage("user", "One"),
                ChatMessage("assistant", "First"),
                ChatMessage("user", "Two"),
                ChatMessage("assistant", "Second"),
            ),
        )
        tokenized = self.template.encode_conversation(record, self.encoder)
        active = [
            tokenized.token_ids[index + 1]
            for index, enabled in enumerate(tokenized.target_mask)
            if enabled
        ]
        expected = (
            self.encoder.encode("First")
            + [60000]
            + self.encoder.encode("Second")
            + [60000]
        )
        self.assertEqual(active, expected)

    def test_generation_prompt_preserves_assistant_history(self) -> None:
        messages = (
            ChatMessage("user", "One"),
            ChatMessage("assistant", "First"),
            ChatMessage("user", "Two"),
        )
        tokens = self.template.encode_generation_prompt(messages, self.encoder)
        self.assertIn(60000, tokens[1:])
        suffix = tuple(self.encoder.encode("\nAssistant:\n"))
        self.assertEqual(tokens[-len(suffix):], suffix)


if __name__ == "__main__":
    unittest.main()
