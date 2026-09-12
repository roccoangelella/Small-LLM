"""Durability contracts for the long production run: drain, retention, auto-resume."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from dataset.src.checkpoint_sequence import (
    find_latest_complete_checkpoint,
    prune_checkpoints,
)
from dataset.src.joint_checkpoint import CheckpointCoordinator
from trainer.cli import main


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import moe_production as production  # noqa: E402


class _TrainerState:
    def __init__(self, step: int) -> None:
        self.step = step

    def state_dict(self) -> dict[str, object]:
        return {"global_step": self.step, "consumed_tokens": self.step * 1_024}

    def load_state_dict(self, state) -> None:
        self.step = int(state["global_step"])


def _coordinator(root: Path) -> CheckpointCoordinator:
    return CheckpointCoordinator(root, configuration_hash="c", source_hash="s", schema_hash="h")


def _write_checkpoint(root: Path, step: int) -> Path:
    return _coordinator(root).save(
        checkpoint_id=f"step-{step:08d}",
        trainer=_TrainerState(step),
        pipeline_state={"last_consumed_block_id": step - 1, "gradient_accumulation_position": 0},
        optimizer_step_complete=True,
    )


def _steps_on_disk(root: Path) -> list[str]:
    return sorted(path.name for path in root.iterdir() if path.is_dir())


class _Metrics:
    def __init__(self, step: int) -> None:
        self.step = step

    def as_dict(self) -> dict[str, object]:
        return {"step": self.step, "loss": 1.0}


class _Engine:
    def __init__(self) -> None:
        self.global_step = 0

    def evaluate(self, batches, *, maximum_batches=None):
        list(batches)
        # A monotonically worsening held-out loss keeps the first update the best one.
        return {"loss": 1.0 + self.global_step, "perplexity": 2.0,
                "target_tokens": 4, "blocks": maximum_batches or 1}


class _ValidationReader:
    def iter_from_start(self, maximum_blocks=None):
        return iter((object(),))


class _Session:
    """Fake session whose step costs wall time and whose checkpoints are real."""

    def __init__(self, engine: _Engine, *, step_seconds: float = 0.0,
                 checkpoint_root: Path | None = None) -> None:
        self.engine = engine
        self.step_seconds = step_seconds
        self.checkpoint_root = checkpoint_root
        self.saved: list[str] = []

    def step(self) -> _Metrics:
        if self.step_seconds:
            time.sleep(self.step_seconds)
        self.engine.global_step += 1
        return _Metrics(self.engine.global_step)

    def save_checkpoint(self, coordinator, checkpoint_id: str, **kwargs):
        self.saved.append(checkpoint_id)
        if self.checkpoint_root is None:
            return f"/tmp/{checkpoint_id}"
        return _write_checkpoint(self.checkpoint_root, self.engine.global_step)


def _run_cli(argv: list[str], session: _Session, engine: _Engine,
             *, checkpoint_every_steps: int = 0) -> str:
    trainer_config = SimpleNamespace(
        evaluation_every_steps=0,
        checkpoint_every_steps=checkpoint_every_steps,
    )
    setup_result = (object(), trainer_config, engine, session, object())
    stream = io.StringIO()
    with (
        patch("trainer.cli.setup", return_value=setup_result),
        patch("trainer.cli.configure_remote_publication", return_value=None),
        patch("trainer.cli.configure_wandb", return_value=None),
        patch("trainer.cli.torch.cuda.is_available", return_value=False),
        contextlib.redirect_stdout(stream),
    ):
        assert main(argv) == 0
    return stream.getvalue()


def _events(output: str, name: str) -> list[dict[str, object]]:
    found = []
    for line in output.splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and isinstance(value.get(name), dict):
            found.append(value[name])
    return found


class DrainOnWallClockTests(unittest.TestCase):
    def test_budget_exhaustion_checkpoints_and_exits_zero(self) -> None:
        engine = _Engine()
        session = _Session(engine, step_seconds=0.01)
        with tempfile.TemporaryDirectory() as directory:
            output = _run_cli(
                ["--dataset-dir", directory, "--checkpoint-dir", directory,
                 "--steps", "500", "--max-wall-seconds", "0.05"],
                session, engine,
            )
        drained = _events(output, "drained")
        self.assertEqual(len(drained), 1)
        self.assertEqual(drained[0]["reason"], "max_wall_seconds")
        self.assertEqual(drained[0]["checkpoint_id"], f"step-{engine.global_step:08d}")
        self.assertLess(engine.global_step, 500)
        self.assertGreater(engine.global_step, 0)
        self.assertEqual(session.saved, [f"step-{engine.global_step:08d}"])
        self.assertEqual(json.loads(output.splitlines()[-1]).keys(), {"drained"})

    def test_drain_on_a_checkpoint_boundary_does_not_save_twice(self) -> None:
        engine = _Engine()
        session = _Session(engine, step_seconds=0.01)
        with tempfile.TemporaryDirectory() as directory:
            _run_cli(
                ["--dataset-dir", directory, "--checkpoint-dir", directory,
                 "--steps", "500", "--max-wall-seconds", "0.05",
                 "--checkpoint-every-steps", "1"],
                session, engine, checkpoint_every_steps=1,
            )
        self.assertLess(engine.global_step, 500)
        self.assertEqual(session.saved, sorted(set(session.saved)))
        self.assertEqual(len(session.saved), engine.global_step)

    def test_zero_budget_runs_every_requested_step(self) -> None:
        engine = _Engine()
        session = _Session(engine)
        with tempfile.TemporaryDirectory() as directory:
            output = _run_cli(
                ["--dataset-dir", directory, "--checkpoint-dir", directory, "--steps", "4"],
                session, engine,
            )
        self.assertEqual(engine.global_step, 4)
        self.assertEqual(_events(output, "drained"), [])


class LocalRetentionTests(unittest.TestCase):
    def test_keep_last_retains_milestones_best_and_the_resume_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in range(1, 8):
                _write_checkpoint(root, step)
            removed = prune_checkpoints(
                root, keep_last=2, protected={"step-00000001", "step-00000003"},
                milestone_every_steps=4,
            )
            self.assertEqual(removed, ["step-00000005", "step-00000002"])
            self.assertEqual(
                _steps_on_disk(root),
                ["step-00000001", "step-00000003", "step-00000004",
                 "step-00000006", "step-00000007"],
            )

    def test_zero_keeps_every_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in range(1, 4):
                _write_checkpoint(root, step)
            self.assertEqual(prune_checkpoints(root, keep_last=0), [])
            self.assertEqual(len(_steps_on_disk(root)), 3)

    def test_an_incomplete_newest_checkpoint_blocks_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in range(1, 4):
                _write_checkpoint(root, step)
            (root / "step-00000009").mkdir()
            self.assertEqual(prune_checkpoints(root, keep_last=1), [])
            self.assertEqual(len(_steps_on_disk(root)), 4)

    def test_trainer_prunes_behind_new_checkpoints_and_keeps_milestones(self) -> None:
        engine = _Engine()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = _Session(engine, checkpoint_root=root)
            output = _run_cli(
                ["--dataset-dir", directory, "--checkpoint-dir", directory, "--steps", "5",
                 "--checkpoint-every-steps", "1", "--keep-last-checkpoints", "2",
                 "--milestone-every-steps", "2"],
                session, engine, checkpoint_every_steps=1,
            )
            self.assertEqual(
                _steps_on_disk(root),
                ["step-00000002", "step-00000004", "step-00000005"],
            )
        retention = _events(output, "checkpoint_retention")
        self.assertEqual([event["removed"] for event in retention],
                         [["step-00000001"], ["step-00000003"]])

    def test_trainer_never_deletes_the_published_best_checkpoint(self) -> None:
        engine = _Engine()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = _Session(engine, checkpoint_root=root)
            argv = [
                "--dataset-dir", directory, "--checkpoint-dir", directory, "--steps", "4",
                "--checkpoint-every-steps", "1", "--keep-last-checkpoints", "1",
                "--validation-blocks", "1", "--evaluation-every-steps", "1",
                "--best-model-repo", "private/best", "--best-model-recreate",
                "--wandb-run-id", "moe-prod-001",
            ]
            trainer_config = SimpleNamespace(evaluation_every_steps=1, checkpoint_every_steps=1)
            with (
                patch("trainer.cli.setup",
                      return_value=(object(), trainer_config, engine, session, object())),
                patch("trainer.cli.validation_reader", return_value=_ValidationReader()),
                patch("trainer.cli.configure_remote_publication", return_value=None),
                patch("trainer.cli.configure_wandb", return_value=None),
                patch("trainer.cli.torch.cuda.is_available", return_value=False),
                patch("trainer.cli.get_dedicated_best_metric", return_value=None),
                patch("trainer.cli.publish_dedicated_best_model",
                      return_value={"published": True}) as publish,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(main(argv), 0)
            self.assertEqual(publish.call_count, 1)
            self.assertEqual(_steps_on_disk(root), ["step-00000001", "step-00000004"])

    def test_trainer_never_deletes_the_resume_source(self) -> None:
        engine = _Engine()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_checkpoint(root, 1)
            engine.global_step = 1
            session = _Session(engine, checkpoint_root=root)
            _run_cli(
                ["--dataset-dir", directory, "--checkpoint-dir", directory, "--steps", "3",
                 "--checkpoint-every-steps", "1", "--keep-last-checkpoints", "1",
                 "--resume", "step-00000001"],
                session, engine, checkpoint_every_steps=1,
            )
            self.assertEqual(
                _steps_on_disk(root), ["step-00000001", "step-00000004"],
            )


class AutoResumeTests(unittest.TestCase):
    def _request(self, root: Path, **overrides) -> production.ProductionRequest:
        values = {
            "run_id": "moe-prod-001",
            "dataset_dir": str(root / "dataset"),
            "total_steps": 10,
            "precision": "fp16",
            "microbatch_size": 8,
            "source_commit": "a" * 40,
        }
        values.update(overrides)
        return production.ProductionRequest(**values)

    def test_absent_resume_starts_fresh_with_the_absolute_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = self._request(root)
            plan = production.resolve_resume(request, run_root=root / "runs")
            self.assertEqual(plan, {"resume": None, "completed_steps": 0, "remaining_steps": 10})
            command = production.build_training_command(request, run_root=root / "runs")
            self.assertNotIn("--resume", command)
            self.assertEqual(command[command.index("--steps") + 1], "10")

    def test_latest_resolves_the_newest_complete_checkpoint_and_remaining_steps(self) -> None:
        for resume in ("latest", None, ""):
            with self.subTest(resume=resume), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                request = self._request(root, resume=resume)
                checkpoints = production.checkpoint_dir(request, run_root=root / "runs")
                checkpoints.mkdir(parents=True)
                for step in (2, 6):
                    _write_checkpoint(checkpoints, step)
                (checkpoints / "step-00000008").mkdir()
                plan = production.resolve_resume(request, run_root=root / "runs")
                self.assertEqual(plan["resume"], "step-00000006")
                self.assertEqual(plan["remaining_steps"], 4)
                command = production.build_training_command(request, run_root=root / "runs")
                self.assertEqual(command[command.index("--resume") + 1], "step-00000006")
                self.assertEqual(command[command.index("--steps") + 1], "4")

    def test_an_explicit_checkpoint_id_is_honoured(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = self._request(root, resume="step-00000003")
            plan = production.resolve_resume(request, run_root=root / "runs")
            self.assertEqual(plan, {"resume": "step-00000003", "completed_steps": 3,
                                    "remaining_steps": 7})

    def test_a_finished_run_is_not_relaunched(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = self._request(root)
            checkpoints = production.checkpoint_dir(request, run_root=root / "runs")
            checkpoints.mkdir(parents=True)
            _write_checkpoint(checkpoints, 10)
            commits = []
            result = production.run_provider_payload(
                production.request_payload(request),
                run_root=root / "runs",
                repo_root=ROOT,
                volume_commit=lambda: commits.append(1),
                popen_factory=_forbidden_popen,
            )
            self.assertEqual(result["steps_launched"], 0)
            self.assertEqual(result["resumed_from"], "step-00000010")
            self.assertEqual(len(commits), 1)


def _forbidden_popen(*args, **kwargs):
    raise AssertionError("a completed run must not launch the trainer again")


class _Process:
    def __init__(self, lines: list[str], code: int = 0) -> None:
        self.stdout = io.StringIO("".join(line + "\n" for line in lines))
        self.code = code

    def wait(self) -> int:
        return self.code

    def kill(self) -> None:
        raise AssertionError("the child must not be killed on a clean run")


class VolumeCommitPerCheckpointTests(unittest.TestCase):
    def test_every_checkpoint_event_commits_the_run_volume(self) -> None:
        lines = [
            json.dumps({"step": 1, "loss": 2.0}),
            json.dumps({"local_checkpoint": {"checkpoint_id": "step-00001000"}}),
            "not json at all",
            json.dumps({"local_checkpoint": {"checkpoint_id": "step-00002000"}}),
            json.dumps({"checkpoint_id": "step-00002000"}),
            json.dumps({"drained": {"checkpoint_id": "step-00002000",
                                    "reason": "max_wall_seconds"}}),
        ]
        commits: list[int] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = production.ProductionRequest(
                run_id="moe-prod-001", dataset_dir=str(root / "dataset"), total_steps=5_000,
                precision="fp16", microbatch_size=8, source_commit="a" * 40,
                max_wall_seconds=60.0,
            )
            result = production.run_provider_payload(
                production.request_payload(request),
                run_root=root / "runs",
                repo_root=ROOT,
                volume_commit=lambda: commits.append(len(commits)),
                popen_factory=lambda *args, **kwargs: _Process(lines),
            )
        self.assertEqual(len(commits), 3)
        self.assertEqual(result["committed_checkpoints"], ["step-00001000", "step-00002000"])
        self.assertEqual(result["status"], "drained")
        self.assertEqual(result["drained"]["reason"], "max_wall_seconds")
        self.assertEqual(result["steps_requested"], 5_000)

    def test_a_failing_child_is_not_reported_as_a_complete_segment(self) -> None:
        commits: list[int] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = production.ProductionRequest(
                run_id="moe-prod-001", dataset_dir=str(root / "dataset"), total_steps=10,
                precision="fp16", microbatch_size=8, source_commit="a" * 40,
            )
            with self.assertRaisesRegex(RuntimeError, "exited with status 9"):
                production.run_provider_payload(
                    production.request_payload(request),
                    run_root=root / "runs",
                    repo_root=ROOT,
                    volume_commit=lambda: commits.append(len(commits)),
                    popen_factory=lambda *args, **kwargs: _Process([], code=9),
                )
        self.assertEqual(commits, [])

    def test_the_child_is_started_as_a_streaming_pipe(self) -> None:
        seen: dict[str, object] = {}

        def factory(command, **kwargs):
            seen.update(kwargs)
            seen["command"] = command
            return _Process([])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = production.ProductionRequest(
                run_id="moe-prod-001", dataset_dir=str(root / "dataset"), total_steps=10,
                precision="fp16", microbatch_size=8, source_commit="a" * 40,
            )
            production.run_provider_payload(
                production.request_payload(request),
                run_root=root / "runs",
                repo_root=ROOT,
                popen_factory=factory,
            )
        self.assertEqual(seen["stdout"], subprocess.PIPE)
        self.assertEqual(seen["stderr"], subprocess.STDOUT)
        self.assertIs(seen["text"], True)
        self.assertEqual(seen["cwd"], str(ROOT))


if __name__ == "__main__":
    unittest.main()
