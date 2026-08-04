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

    def test_remote_publication_requires_drive_manifest(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args([*_BASE, "--remote-publish-every-steps", "50"])

    def test_remote_publication_arguments_are_retained(self) -> None:
        args = parse_args(
            [
                *_BASE,
                "--remote-publish-every-steps",
                "50",
                "--remote-drive-manifest",
                "/tmp/drive_manifest.json",
                "--remote-checkpoint-repo",
                "owner/private-checkpoints",
            ]
        )
        self.assertEqual(args.remote_publish_every_steps, 50)
        self.assertEqual(str(args.remote_drive_manifest), "/tmp/drive_manifest.json")
        self.assertEqual(args.remote_checkpoint_repo, "owner/private-checkpoints")

    def test_remote_publication_cadence_cannot_be_negative(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args([*_BASE, "--remote-publish-every-steps", "-1"])

    def test_wandb_defaults_are_network_free_and_use_project_name(self) -> None:
        args = parse_args(list(_BASE))
        self.assertEqual(args.wandb_mode, "disabled")
        self.assertEqual(args.wandb_project, "Small-LLM")
        self.assertEqual(args.wandb_resume, "never")

    def test_wandb_resume_requires_run_id(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    *_BASE,
                    "--resume",
                    "step-00000025",
                    "--wandb-mode",
                    "online",
                ]
            )

    def test_training_resume_forces_matching_wandb_resume(self) -> None:
        args = parse_args(
            [
                *_BASE,
                "--resume",
                "step-00000025",
                "--wandb-mode",
                "online",
                "--wandb-run-id",
                "qualification-001",
            ]
        )
        self.assertEqual(args.wandb_resume, "must")

    def test_disabled_wandb_rejects_resume_policy(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args([*_BASE, "--wandb-resume", "allow"])


if __name__ == "__main__":
    unittest.main()
