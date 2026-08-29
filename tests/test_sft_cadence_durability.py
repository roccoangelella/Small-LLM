from __future__ import annotations

import unittest

from post_training.sft.train_cli import _cadence_actions, build_parser


class SFTCadenceDurabilityTests(unittest.TestCase):
    def test_trainer_accepts_explicit_rolling_remote_retention(self) -> None:
        args = build_parser().parse_args(
            [
                "--dataset-dir",
                "/tmp/data",
                "--checkpoint-dir",
                "/tmp/checkpoints",
                "--sft-run-id",
                "run",
                "--remote-rolling-latest-only",
            ]
        )
        self.assertTrue(args.remote_rolling_latest_only)

    def test_shared_boundary_persists_and_publishes_before_evaluation(self) -> None:
        self.assertEqual(
            _cadence_actions(
                250,
                checkpoint_every_steps=250,
                remote_publish_every_steps=250,
                evaluation_every_steps=250,
            ),
            ("checkpoint", "publish", "evaluate"),
        )

    def test_remote_publication_forces_local_checkpoint_first(self) -> None:
        self.assertEqual(
            _cadence_actions(
                100,
                checkpoint_every_steps=250,
                remote_publish_every_steps=100,
                evaluation_every_steps=50,
            ),
            ("checkpoint", "publish", "evaluate"),
        )

    def test_evaluation_only_boundary_does_not_create_checkpoint(self) -> None:
        self.assertEqual(
            _cadence_actions(
                50,
                checkpoint_every_steps=250,
                remote_publish_every_steps=250,
                evaluation_every_steps=50,
            ),
            ("evaluate",),
        )


if __name__ == "__main__":
    unittest.main()
