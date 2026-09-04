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


def test_chat_completion_gate_accepts_100m_10b_wsqd_horizon() -> None:
    config = {
        "schedule": "wsqd",
        "schedule_anchor_tokens": 2_031_616_000,
        "cooldown_start_tokens": 9_600_008_192,
        "decay_tokens": 399_998_976,
        "settle_tokens": 299_991_040,
        "settle_lr_ratio": 0.3333333333333333,
        "base_power": 0.5,
        "microbatch_size": 16,
        "source_checkpoint_id": "step-00015500",
        "run_id": "100m-10b-deep-decay-from-step15500",
    }
    assert chat._completed_schedule_targets(config) == 10_000_007_168


def test_chat_completion_gate_rejects_schedule_without_known_terminal_horizon() -> None:
    with pytest.raises(RuntimeError, match="unsupported completion schedule"):
        chat._completed_schedule_targets({"schedule": "constant"})
