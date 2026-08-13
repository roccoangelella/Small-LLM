"""Source-level guardrails for the Modal CPU-before-H100 rolling dataset contract."""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ModalRollingDispatchTests(unittest.TestCase):
    def test_incremental_producer_and_supervised_stage_precede_h100_spawn(self) -> None:
        source = (ROOT / "modal" / "launch.py").read_text(encoding="utf-8")

        producer = source.index("producer_call = produce_rolling_dataset_remote.spawn")
        stage = source.index("stage_call = stage_rolling_dataset_remote.spawn")
        supervise = source.index("await_stage_with_producer(stage_call, producer_call)")
        authorize = source.index('staged.get("h100_dispatch_allowed") is not True')
        spawn = source.index("result = train_rolling_remote.with_options")
        self.assertLess(producer, stage)
        self.assertLess(stage, supervise)
        self.assertLess(supervise, authorize)
        self.assertLess(authorize, spawn)
        self.assertIn('"h100_allocated": False', source)
        self.assertNotIn("stage_rolling_dataset_remote.remote", source)

    def test_dataset_producer_is_cpu_only_and_h100_has_no_automatic_retry(self) -> None:
        source = (ROOT / "modal" / "launch.py").read_text(encoding="utf-8")

        producer_decorator_start = source.rindex(
            "@app.function(", 0, source.index("def produce_rolling_dataset_remote")
        )
        producer_function_start = source.index("def produce_rolling_dataset_remote")
        producer_decorator = source[producer_decorator_start:producer_function_start]
        self.assertIn("cpu=4.0", producer_decorator)
        self.assertIn("memory=8192", producer_decorator)
        self.assertNotIn("gpu=", producer_decorator)

        h100_decorator_start = source.rindex(
            "@app.function(", 0, source.index("def train_rolling_remote")
        )
        h100_function_start = source.index("def train_rolling_remote")
        h100_decorator = source[h100_decorator_start:h100_function_start]
        self.assertIn("gpu=DEFAULT_GPU", h100_decorator)
        self.assertNotIn("retries=", h100_decorator)

    def test_cpu_stage_uses_checkpoint_aligned_incremental_lead_window(self) -> None:
        source = (ROOT / "modal" / "rolling_dataset.py").read_text(encoding="utf-8")

        self.assertIn("next_unconsumed_block", source)
        self.assertIn("stage_incremental_window", source)
        self.assertIn('start_block_id=int(cursor["next_block_id"])', source)
        self.assertIn("checkpoint advanced after CPU dataset staging", source)
        self.assertIn("verify_incremental_stage", source)

    def test_modal_microbatch_probe_does_not_start_one_gib_successor_prefetch(self) -> None:
        source = (ROOT / "trainer" / "cli_setup.py").read_text(encoding="utf-8")

        self.assertIn('os.environ.get("SMALL_LLM_MODAL_ROLLING_DATASET") == "1"', source)
        self.assertIn('getattr(args, "wandb_mode", "disabled") == "disabled"', source)
        self.assertIn("return None", source)


if __name__ == "__main__":
    unittest.main()
