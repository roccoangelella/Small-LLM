from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from tests.trainer_fixtures import TinyLM, batch
from trainer.config import TrainerConfig
from trainer.step import _fp16_overflow_retry_limit, train_step


class _ScaledLoss:
    def __init__(self, loss: torch.Tensor) -> None:
        self.loss = loss

    def backward(self) -> None:
        self.loss.backward()


class _SimulatedGradScaler:
    """CPU test double that skips a chosen number of optimizer steps."""

    def __init__(self, *, scale: float, overflow_attempts: int) -> None:
        self._scale = float(scale)
        self._remaining = int(overflow_attempts)
        self._pending_overflow = False

    def is_enabled(self) -> bool:
        return True

    def get_scale(self) -> float:
        return self._scale

    def get_backoff_factor(self) -> float:
        return 0.5

    def scale(self, loss: torch.Tensor) -> _ScaledLoss:
        return _ScaledLoss(loss)

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        del optimizer

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        self._pending_overflow = self._remaining > 0
        if self._pending_overflow:
            self._remaining -= 1
        else:
            optimizer.step()

    def update(self, new_scale: float | None = None) -> None:
        if new_scale is not None:
            self._scale = float(new_scale)
        elif self._pending_overflow:
            self._scale *= self.get_backoff_factor()


class _Scheduler:
    def __init__(self) -> None:
        self.committed_tokens = 0

    def prepare_step(self, next_tokens: int) -> float:
        self.prepared_tokens = next_tokens
        return 3e-4

    def commit(self, tokens: int) -> None:
        self.committed_tokens = tokens


class FP16OverflowCalibrationTests(unittest.TestCase):
    def test_retry_limit_reaches_scale_one(self):
        scaler = _SimulatedGradScaler(scale=2048.0, overflow_attempts=0)
        self.assertEqual(_fp16_overflow_retry_limit(scaler, 3), 11)

    def test_block_can_survive_more_than_configured_retries(self):
        torch.manual_seed(1)
        model = TinyLM()
        optimizer = torch.optim.SGD(model.parameters(), lr=3e-4)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        engine = SimpleNamespace(
            device=torch.device("cpu"),
            config=TrainerConfig(
                precision="fp32",
                microbatch_size=1,
                weight_decay=0.0,
                max_overflow_retries=3,
            ),
            model=model,
            optimizer=optimizer,
            scheduler=_Scheduler(),
            scaler=_SimulatedGradScaler(scale=2048.0, overflow_attempts=4),
            consumed_tokens=0,
            global_step=0,
            overflow_events=0,
        )

        metrics = train_step(engine, batch(0))

        self.assertEqual(metrics.step, 1)
        self.assertEqual(metrics.overflow_retries, 4)
        self.assertEqual(metrics.overflow_events_total, 4)
        self.assertEqual(metrics.grad_scaler_scale, 128.0)
        self.assertEqual(engine.consumed_tokens, 6)
        self.assertEqual(engine.scheduler.committed_tokens, 6)
        self.assertTrue(
            any(
                not torch.equal(parameter.detach(), original)
                for parameter, original in zip(model.parameters(), before, strict=True)
            )
        )


if __name__ == "__main__":
    unittest.main()
