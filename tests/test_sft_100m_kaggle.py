from __future__ import annotations

import unittest


class SFT100M2BKaggleTests(unittest.TestCase):
    def test_ten_percent_budget(self) -> None:
        self.assertEqual(2_001_000_448 * 10 // 100, 200_100_044)


if __name__ == "__main__":
    unittest.main()
