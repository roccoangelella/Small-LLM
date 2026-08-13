"""Local supervision for concurrent Modal CPU dataset calls.

This module deliberately has no Modal imports so its failure ordering can be
unit-tested without constructing an App.  It supervises the producer while the
CPU staging gate waits for the minimum READY lead.  A producer failure therefore
cancels staging and surfaces before any H100 function is spawned.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _cancel(call: object | None) -> None:
    if call is None:
        return
    cancel = getattr(call, "cancel", None)
    if not callable(cancel):
        return
    try:
        cancel(terminate_containers=True)
    except Exception:
        # Cancellation is cleanup after another failure and must never mask the
        # original producer/staging exception.
        pass


def _poll_completed(call: object) -> tuple[bool, Any]:
    get = getattr(call, "get", None)
    if not callable(get):
        raise TypeError("Modal call handle has no get method")
    try:
        return True, get(timeout=0)
    except TimeoutError:
        return False, None


def await_stage_with_producer(
    stage_call: object,
    producer_call: object | None,
    *,
    poll_seconds: float = 5.0,
) -> tuple[dict[str, object], dict[str, object] | None]:
    """Wait for CPU staging while failing promptly if the producer fails.

    ``FunctionCall.get(timeout=0)`` is Modal's documented non-blocking poll.
    The stage call is allowed to block for ``poll_seconds`` at a time so this
    loop does not generate high-frequency control-plane traffic.

    Returns the staging result plus an already-completed producer result when
    production happened to finish before staging.  If production is still
    active when staging succeeds, the second element is ``None`` and the caller
    retains the original producer handle for later observation.
    """

    if poll_seconds <= 0:
        raise ValueError("CPU supervision poll_seconds must be positive")
    stage_get = getattr(stage_call, "get", None)
    if not callable(stage_get):
        raise TypeError("Modal stage call handle has no get method")

    producer_finished = producer_call is None
    producer_result: dict[str, object] | None = None

    while True:
        try:
            raw_stage = stage_get(timeout=poll_seconds)
        except TimeoutError:
            if producer_call is not None and not producer_finished:
                try:
                    done, raw_producer = _poll_completed(producer_call)
                except BaseException:
                    _cancel(stage_call)
                    raise
                if done:
                    if not isinstance(raw_producer, Mapping):
                        _cancel(stage_call)
                        raise RuntimeError("CPU dataset producer returned a non-object result")
                    producer_result = dict(raw_producer)
                    producer_finished = True
            continue
        except BaseException:
            if producer_call is not None and not producer_finished:
                _cancel(producer_call)
            raise

        if not isinstance(raw_stage, Mapping):
            if producer_call is not None and not producer_finished:
                _cancel(producer_call)
            raise RuntimeError("CPU dataset staging returned a non-object result")

        # Close the race where the producer failed while the stage result was
        # becoming available.  A still-running producer raises TimeoutError and
        # is therefore acceptable; a completed exception is propagated before
        # the caller can dispatch an H100.
        if producer_call is not None and not producer_finished:
            try:
                done, raw_producer = _poll_completed(producer_call)
            except BaseException:
                raise
            if done:
                if not isinstance(raw_producer, Mapping):
                    raise RuntimeError("CPU dataset producer returned a non-object result")
                producer_result = dict(raw_producer)
                producer_finished = True

        return dict(raw_stage), producer_result


__all__ = ["await_stage_with_producer"]
