from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from dataset.src.joint_checkpoint import CheckpointCoordinator
from trainer.state import load_trainer_state_file, save_engine_checkpoint_state


class _StreamingTrainer:
    def __init__(self) -> None:
        self.saved = False
        self.loaded = False
        self.value = 7

    def state_dict(self):
        raise AssertionError("streaming trainer must not use coordinator pickle state_dict")

    def load_state_dict(self, state):
        del state
        raise AssertionError("streaming trainer must not use coordinator pickle load_state_dict")

    def save_checkpoint_state(self, path: Path) -> None:
        self.saved = True
        torch.save({"value": self.value}, path)

    def load_checkpoint_state(self, path: Path) -> None:
        self.loaded = True
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.value = int(payload["value"])


class _StateComponent:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def state_dict(self) -> dict[str, object]:
        return self.payload


class _Config:
    def as_dict(self) -> dict[str, object]:
        return {"precision": "fp32"}


class CheckpointStreamingTests(unittest.TestCase):
    def test_coordinator_prefers_streaming_trainer_io(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trainer = _StreamingTrainer()
            coordinator = CheckpointCoordinator(
                root / "checkpoints",
                configuration_hash="cfg",
                source_hash="src",
                schema_hash="schema",
            )
            coordinator.save(
                checkpoint_id="step-1",
                trainer=trainer,
                pipeline_state={
                    "gradient_accumulation_position": 0,
                    "last_consumed_block_id": 0,
                },
                optimizer_step_complete=True,
            )
            self.assertTrue(trainer.saved)
            trainer.value = 0
            pipeline = coordinator.load("step-1", trainer)
            self.assertTrue(trainer.loaded)
            self.assertEqual(trainer.value, 7)
            self.assertEqual(pipeline["last_consumed_block_id"], 0)

    def test_trainer_state_loader_accepts_streamed_and_legacy_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            streamed = root / "streamed.pkl"
            legacy = root / "legacy.pkl"
            torch.save({"version": 1, "value": torch.tensor([3])}, streamed)
            with legacy.open("wb") as handle:
                pickle.dump({"version": 1, "value": 4}, handle, protocol=pickle.HIGHEST_PROTOCOL)
            streamed_state = load_trainer_state_file(streamed, map_location="cpu")
            legacy_state = load_trainer_state_file(legacy, map_location="cpu")
            self.assertEqual(int(streamed_state["value"][0]), 3)
            self.assertEqual(legacy_state["value"], 4)

    def test_streamed_engine_save_does_not_build_cpu_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trainer_state.pkl"
            engine = SimpleNamespace(
                config=_Config(),
                model=_StateComponent({"weight": torch.tensor([1.0])}),
                optimizer=_StateComponent({"state": {}, "param_groups": []}),
                scheduler=_StateComponent({}),
                scaler=_StateComponent({}),
                global_step=2,
                consumed_tokens=11,
                overflow_events=0,
                best_validation_loss=None,
                device=torch.device("cpu"),
            )
            with mock.patch("trainer.state.cpu_tree", side_effect=AssertionError("cpu clone")):
                save_engine_checkpoint_state(engine, path)
            state = load_trainer_state_file(path, map_location="cpu")
            self.assertEqual(state["global_step"], 2)
            self.assertEqual(state["consumed_tokens"], 11)
            self.assertTrue(torch.equal(state["model"]["weight"], torch.tensor([1.0])))


if __name__ == "__main__":
    unittest.main()
