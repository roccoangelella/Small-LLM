"""The production command takes its learning-rate schedule from the corpus run contract."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import moe_production as production
from MOE_model.__main__ import parse_args
from dataset.incremental_frontier import standard_wsd_plan

LIVE_TRAINER = {
    "decay_tokens": 20000014336, "decay_updates": 152588, "full_block_target_tokens": 131072,
    "minimum_lr_ratio": 0.1, "passes": 1, "planned_target_tokens": 100000071680,
    "schedule": "wsd", "stable_tokens": 75000053760, "stable_updates": 572205,
    "steps": 762940, "validation_blocks": 16, "warmup_tokens": 5000003584, "warmup_updates": 38147,
}


def _write_contract(root: Path, **fields) -> Path:
    contract = {"version": 1, "run_id": "test", "schema_version": 2, "context_length": 2048,
                "sequences_per_block": 64, **fields}
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    return root


def _request(dataset_dir: Path, **overrides) -> production.ProductionRequest:
    values = {"run_id": "moe-prod", "dataset_dir": str(dataset_dir), "total_steps": 762_940,
              "precision": "bf16", "microbatch_size": 64, "source_commit": "a" * 40}
    values.update(overrides)
    return production.ProductionRequest(**values)


def _value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


class TestProductionSchedule(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_live_contract_schedule_is_passed_verbatim(self) -> None:
        data = _write_contract(self.root / "data", trainer=LIVE_TRAINER)
        command = production.build_training_command(_request(data), run_root=self.root / "runs")
        self.assertEqual(_value(command, "--schedule"), "wsd")
        self.assertEqual(_value(command, "--warmup-tokens"), "5000003584")
        self.assertEqual(_value(command, "--stable-tokens"), "75000053760")
        self.assertEqual(_value(command, "--decay-tokens"), "20000014336")
        self.assertEqual(_value(command, "--minimum-lr-ratio"), "0.10000000000000001")
        self.assertEqual(_value(command, "--validation-blocks"), "16")
        self.assertEqual(_value(command, "--evaluation-every-steps"), "1000")
        for flag, expected in (("--learning-rate", "3e-4"), ("--muon-update-rms", "0.18"),
                               ("--muon-weight-decay", "0.1"), ("--max-grad-norm", "1.0")):
            self.assertEqual(_value(command, flag), expected)

    def test_geometry_only_contract_derives_the_standard_plan(self) -> None:
        data = _write_contract(self.root / "data", planned_train_blocks=4157)
        schedule = production.resolve_schedule(data)
        expected = standard_wsd_plan(4157, context_length=2048, sequences_per_block=64,
                                     validation_blocks=16)
        for key in ("steps", "warmup_tokens", "stable_tokens", "decay_tokens", "minimum_lr_ratio"):
            self.assertEqual(schedule[key], expected[key])
        self.assertEqual(schedule["validation_blocks"], 16)

    def test_missing_contract_fails_closed(self) -> None:
        with self.assertRaises(FileNotFoundError):
            production.build_training_command(_request(self.root / "absent"), run_root=self.root)

    def test_constant_schedule_contract_is_refused(self) -> None:
        data = _write_contract(self.root / "data", trainer={**LIVE_TRAINER, "schedule": "constant"})
        with self.assertRaises(ValueError):
            production.resolve_schedule(data)

    def test_total_steps_cannot_exceed_the_plan(self) -> None:
        data = _write_contract(self.root / "data", trainer=LIVE_TRAINER)
        with self.assertRaises(ValueError):
            production.build_training_command(_request(data, total_steps=762_941), run_root=self.root)

    def test_explicit_zero_validation_disables_it(self) -> None:
        data = _write_contract(self.root / "data", trainer=LIVE_TRAINER)
        command = production.build_training_command(
            _request(data, validation_blocks=0), run_root=self.root,
        )
        self.assertEqual(_value(command, "--validation-blocks"), "0")

    def test_trainer_parses_the_scheduled_command(self) -> None:
        data = _write_contract(self.root / "data", trainer=LIVE_TRAINER)
        command = production.build_training_command(_request(data), run_root=self.root)
        args = parse_args(command[command.index("-m") + 2:] + ["--device", "cpu"])
        self.assertEqual(args.schedule, "wsd")
        self.assertEqual(args.warmup_tokens, 5000003584)
        self.assertEqual(args.decay_tokens, 20000014336)
        self.assertEqual(args.muon_update_rms, 0.18)
        self.assertEqual(args.evaluation_every_steps, 1000)


if __name__ == "__main__":
    unittest.main()
