"""Network-free tests for Modal CPU producer/stager supervision."""

from __future__ import annotations

import importlib.util
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
        assert terminate_containers is True
        self.cancelled = True


def test_stage_wait_polls_running_producer_then_returns_ready() -> None:
    stage = FakeCall([TimeoutError(), {"status": "ready"}])
    producer = FakeCall([TimeoutError(), TimeoutError()])

    result, producer_result = await_stage_with_producer(stage, producer, poll_seconds=2.0)

    assert result == {"status": "ready"}
    assert producer_result is None
    assert stage.get_timeouts == [2.0, 2.0]
    assert producer.get_timeouts == [0, 0]
    assert producer.cancelled is False


def test_producer_failure_cancels_stage_before_ready() -> None:
    stage = FakeCall([TimeoutError()])
    producer = FakeCall([RuntimeError("producer failed")])

    try:
        await_stage_with_producer(stage, producer, poll_seconds=1.0)
    except RuntimeError as error:
        assert str(error) == "producer failed"
    else:
        raise AssertionError("producer failure was not propagated")

    assert stage.cancelled is True


def test_stage_failure_cancels_running_producer() -> None:
    stage = FakeCall([RuntimeError("stage failed")])
    producer = FakeCall([])

    try:
        await_stage_with_producer(stage, producer)
    except RuntimeError as error:
        assert str(error) == "stage failed"
    else:
        raise AssertionError("stage failure was not propagated")

    assert producer.cancelled is True


def test_completed_producer_result_is_returned_with_ready_stage() -> None:
    stage = FakeCall([{"status": "ready"}])
    producer = FakeCall([{"status": "complete", "producer_complete": True}])

    result, producer_result = await_stage_with_producer(stage, producer)

    assert result == {"status": "ready"}
    assert producer_result == {"status": "complete", "producer_complete": True}


def test_non_object_results_fail_closed() -> None:
    stage = FakeCall(["ready"])
    producer = FakeCall([])

    try:
        await_stage_with_producer(stage, producer)
    except RuntimeError as error:
        assert "non-object" in str(error)
    else:
        raise AssertionError("non-object stage result was accepted")

    assert producer.cancelled is True
