"""Network-free tests for Modal CPU producer/stager supervision."""

from __future__ import annotations

import importlib.util
import unittest
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "small_llm_modal_cpu_supervision",
    ROOT / "modal" / "cpu_supervision.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
await_stage_with_producer = MODULE.await_stage_with_producer


class FakeCall:
    def __init__(self, outcomes):
        self.outcomes = deque(outcomes)
        self.get_timeouts = []
        self.cancelled = False

    def get(self, timeout=None):
        self.get_timeouts.append(timeout)
        if not self.outcomes:
            raise AssertionError("unexpected extra poll")
        outcome = self.outcomes.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def cancel(self, terminate_containers=False):
        if terminate_containers is not True:
            raise AssertionError("cleanup must terminate spawned containers")
        self.cancelled = True


class ModalCpuSupervisionTests(unittest.TestCase):
    def test_stage_wait_polls_running_producer_then_returns_ready(self) -> None:
        stage = FakeCall([TimeoutError(), {"status": "ready"}])
        producer = FakeCall([TimeoutError(), TimeoutError()])

        result, producer_result = await_stage_with_producer(stage, producer, poll_seconds=2.0)

        self.assertEqual(result, {"status": "ready"})
        self.assertIsNone(producer_result)
        self.assertEqual(stage.get_timeouts, [2.0, 2.0])
        self.assertEqual(producer.get_timeouts, [0, 0])
        self.assertFalse(producer.cancelled)

    def test_producer_failure_cancels_stage_before_ready(self) -> None:
        stage = FakeCall([TimeoutError()])
        producer = FakeCall([RuntimeError("producer failed")])

        with self.assertRaisesRegex(RuntimeError, "producer failed"):
            await_stage_with_producer(stage, producer, poll_seconds=1.0)

        self.assertTrue(stage.cancelled)

    def test_stage_failure_cancels_running_producer(self) -> None:
        stage = FakeCall([RuntimeError("stage failed")])
        producer = FakeCall([])

        with self.assertRaisesRegex(RuntimeError, "stage failed"):
            await_stage_with_producer(stage, producer)

        self.assertTrue(producer.cancelled)

    def test_completed_producer_result_is_returned_with_ready_stage(self) -> None:
        stage = FakeCall([{"status": "ready"}])
        producer = FakeCall([{"status": "complete", "producer_complete": True}])

        result, producer_result = await_stage_with_producer(stage, producer)

        self.assertEqual(result, {"status": "ready"})
        self.assertEqual(producer_result, {"status": "complete", "producer_complete": True})

    def test_non_object_results_fail_closed(self) -> None:
        stage = FakeCall(["ready"])
        producer = FakeCall([])

        with self.assertRaisesRegex(RuntimeError, "non-object"):
            await_stage_with_producer(stage, producer)

        self.assertTrue(producer.cancelled)


if __name__ == "__main__":
    unittest.main()
