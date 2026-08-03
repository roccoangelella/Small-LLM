"""CLI safety tests for the corrected trusted GDN-2 launch path."""

from __future__ import annotations

import unittest

from trainer.cli_args import parse_args


_BASE = (
    "--dataset-dir",
    "/tmp/data",
    "--checkpoint-dir",
    "/tmp/checkpoints",
    "--steps",
    "1",
)


class TrainerCLIArgumentTests(unittest.TestCase):
    def test_trusted_fp16_gdn2_defaults_to_chunk_32_and_hybrid_optimizer(self) -> None:
        args = parse_args(list(_BASE))
        self.assertEqual(args.gdn_chunk_size, 32)
        self.assertEqual(args.optimizer, "hybrid_muon_adamw")

    def test_fp32_gdn2_retains_chunk_64_default(self) -> None:
        args = parse_args([*_BASE, "--precision", "fp32"])
        self.assertEqual(args.gdn_chunk_size, 64)

    def test_unqualified_fp16_chunk_requires_diagnostic_override(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args([*_BASE, "--gdn-chunk-size", "64"])
        args = parse_args(
            [
                *_BASE,
                "--gdn-chunk-size",
                "64",
                "--allow-unqualified-gdn2-chunk",
            ]
        )
        self.assertEqual(args.gdn_chunk_size, 64)

    def test_transformer_path_rejects_gdn_chunk_argument(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    *_BASE,
                    "--architecture",
                    "swa_hybrid",
                    "--gdn-chunk-size",
                    "32",
                ]
            )


if __name__ == "__main__":
    unittest.main()
