from __future__ import annotations

import argparse

import pytest

import chat


def test_parse_quantity_accepts_profile_spellings() -> None:
    assert chat._parse_quantity("20M") == 20_000_000
    assert chat._parse_quantity("500m") == 500_000_000
    assert chat._parse_quantity("2B") == 2_000_000_000
    assert chat._parse_quantity("20_000_000") == 20_000_000


def test_parse_quantity_rejects_non_integral_sizes() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        chat._parse_quantity("0")
    with pytest.raises(argparse.ArgumentTypeError):
        chat._parse_quantity("1.5")
    with pytest.raises(argparse.ArgumentTypeError):
        chat._parse_quantity("20Q")


def test_resolve_sft_run_id_is_fail_closed() -> None:
    assert chat._resolve_sft_run_id(20_000_000, 500_000_000) == "20m-500m-sft-s0-001"
    assert chat._resolve_sft_run_id(20_000_000, 2_000_000_000) == "20m-2b-sft-s0-001"
    with pytest.raises(RuntimeError, match="no registered SFT profile"):
        chat._resolve_sft_run_id(100_000_000, 2_000_000_000)


class _FakeTemplate:
    def __init__(self, hard_limit: int = 25) -> None:
        self.hard_limit = hard_limit

    def encode_generation_prompt(self, history, encoding):
        del encoding
        size = len(history) * 10
        if size > self.hard_limit:
            raise ValueError("generation prompt exceeds model context")
        return tuple(range(size))


def test_fit_generation_prompt_drops_oldest_complete_turn() -> None:
    history = ["u1", "a1", "u2"]
    fitted, prompt_ids = chat._fit_generation_prompt(
        history,
        template=_FakeTemplate(),
        encoding=object(),
        max_prompt_tokens=20,
    )
    assert fitted == ["u2"]
    assert len(prompt_ids) == 10


def test_fit_generation_prompt_rejects_single_overlong_message() -> None:
    with pytest.raises(RuntimeError, match="message is too long"):
        chat._fit_generation_prompt(
            ["u1"],
            template=_FakeTemplate(hard_limit=5),
            encoding=object(),
            max_prompt_tokens=5,
        )
