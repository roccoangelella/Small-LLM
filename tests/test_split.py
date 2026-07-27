"""Deterministic, order-independent train/validation assignment."""

from __future__ import annotations

import hashlib
import unittest

from dataset import config
from dataset.src.split import is_validation


def _manual(seed: str, revision: str, filename: str, record_start: int, probability: float) -> bool:
    def component(value: str) -> bytes:
        encoded = value.encode("utf-8")
        return len(encoded).to_bytes(8, "big") + encoded

    material = b"".join(
        (
            component(config.SPLIT_HASH_VERSION),
            component(seed),
            component(revision),
            component(filename),
            record_start.to_bytes(8, "big"),
        )
    )
    draw = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return draw < int(probability * (1 << 64))


class SplitTest(unittest.TestCase):
    def test_matches_manual_hash_and_is_stable(self) -> None:
        args = dict(seed="seed-1", revision="rev-9", filename="part_0", record_start=12345,
                    probability=0.001)
        self.assertEqual(is_validation(**args), _manual(**args))
        self.assertEqual(is_validation(**args), is_validation(**args))

    def test_probability_zero_is_never_validation(self) -> None:
        for start in range(200):
            self.assertFalse(is_validation(seed="s", revision="r", filename="f",
                                           record_start=start, probability=0.0))

    def test_probability_one_is_always_validation(self) -> None:
        for start in range(200):
            self.assertTrue(is_validation(seed="s", revision="r", filename="f",
                                          record_start=start, probability=1.0))

    def test_never_both_and_rate_is_approximately_correct(self) -> None:
        total = 20_000
        validations = sum(
            is_validation(seed="small-llm-climbmix-production-v1", revision="rev",
                          filename="part_x.tokenized.jsonl", record_start=start,
                          probability=0.001)
            for start in range(total)
        )
        # Expected ~0.1 % = 20; allow a generous deterministic spread.
        self.assertGreater(validations, 4)
        self.assertLess(validations, 80)

    def test_assignment_depends_only_on_identity_not_order(self) -> None:
        # Two different calls with the same identity but "processed" in different
        # orders must agree (the function has no order input).
        identity = dict(seed="seed", revision="rev", filename="f", probability=0.5)
        first = is_validation(record_start=100, **identity)
        for other in (50, 200, 1, 9999):
            _ = is_validation(record_start=other, **identity)
        self.assertEqual(first, is_validation(record_start=100, **identity))

    def test_length_prefixing_prevents_separator_ambiguity(self) -> None:
        left = is_validation(
            seed="a:b", revision="c", filename="d", record_start=1, probability=0.5
        )
        right = is_validation(
            seed="a", revision="b:c", filename="d", record_start=1, probability=0.5
        )
        # The canonical byte identities differ even though naive colon joining
        # would produce the same prefix.
        self.assertEqual(
            left,
            _manual("a:b", "c", "d", 1, 0.5),
        )
        self.assertEqual(
            right,
            _manual("a", "b:c", "d", 1, 0.5),
        )


if __name__ == "__main__":
    unittest.main()
