"""Shared exception helpers for the dataset pipeline."""

from __future__ import annotations


class IntentionalCrash(RuntimeError):
    """Raised only when an explicit test/smoke crash is requested via the CLI.

    A process killed after writing binary bytes but before committing the
    checkpoint must recover correctly; this lets tests reproduce that exact
    scenario deterministically without ``SIGKILL``.
    """


def is_intentional_crash(error: BaseException) -> bool:
    """Return True iff ``error`` is the test-only intentional crash signal."""

    return isinstance(error, IntentionalCrash)