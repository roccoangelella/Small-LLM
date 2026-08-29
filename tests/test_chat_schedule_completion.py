from __future__ import annotations

import pytest

import chat
from trainer.fresh_decay import fresh_aggressive_decay_plan


def test_chat_completion_gate_accepts_10pct_fresh_wsqd_horizon() -> None:
    realized_train_targets = 200_099_738
    plan = fresh_aggressive_decay_plan(realized_train_targets)

    assert chat._completed_schedule_targets(plan.trainer_kwargs()) == realized_train_targets


def test_chat_completion_gate_preserves_historical_wsd_horizon() -> None:
    assert chat._completed_schedule_targets(
        {
            "schedule": "wsd",
            "warmup_tokens": 10,
            "stable_tokens": 80,
            "decay_tokens": 10,
        }
    ) == 100


def test_chat_completion_gate_rejects_schedule_without_known_terminal_horizon() -> None:
    with pytest.raises(RuntimeError, match="unsupported completion schedule"):
        chat._completed_schedule_targets({"schedule": "constant"})
