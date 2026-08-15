from __future__ import annotations

import argparse

import pytest

import chat


def test_parse_quantity_accepts_profile_spellings() -> None:
    assert chat._parse_quantity("20M") == 20_000_000
    assert chat._parse_quantity("100M") == 100_000_000
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


def test_resolve_chat_run_is_fail_closed() -> None:
    assert chat._resolve_chat_run(20_000_000, 500_000_000) == (
        "20m-500m-sft-s0-001",
        chat._SOURCE_SFT,
    )
    assert chat._resolve_chat_run(20_000_000, 2_000_000_000) == (
        "20m-2b-sft-s0-001",
        chat._SOURCE_SFT,
    )
    assert chat._resolve_chat_run(100_000_000, 2_000_000_000) == (
        "100m-2b-data-001",
        chat._SOURCE_STABLE_MODEL,
    )
    assert chat._resolve_chat_run(
        100_000_000,
        2_000_000_000,
        prefer_sft=True,
    ) == (
        "100m-2b-sft-s0-001",
        chat._SOURCE_SFT,
    )
    with pytest.raises(RuntimeError, match="no registered chat profile"):
        chat._resolve_chat_run(100_000_000, 10_000_000_000)
    with pytest.raises(RuntimeError, match="no registered SFT chat profile"):
        chat._resolve_chat_run(100_000_000, 10_000_000_000, prefer_sft=True)


def test_sft_flag_is_explicit() -> None:
    default_args = chat._parse_args(["--model_params", "100M", "--num_tokens", "2B"])
    sft_args = chat._parse_args(
        ["--model_params", "100M", "--num_tokens", "2B", "--sft"]
    )
    assert default_args.sft is False
    assert sft_args.sft is True


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


class _SplitUtf8Encoding:
    _TOKENS = {
        1: b"plain ",
        2: b"\xe2",
        3: b"\x82",
        4: b"\xac",
    }

    def decode_single_token_bytes(self, token_id: int) -> bytes:
        return self._TOKENS[token_id]


def test_token_streamer_preserves_utf8_across_token_boundaries() -> None:
    streamer = chat._TokenTextStreamer(_SplitUtf8Encoding())
    assert streamer.push(1) == "plain "
    assert streamer.push(2) == ""
    assert streamer.push(3) == ""
    assert streamer.push(4) == "€"
    assert streamer.finish() == ""
