from __future__ import annotations

import pytest

import chat


def test_registered_rsft_default_is_expanded_e3_run() -> None:
    assert chat._resolve_chat_run(
        100_000_000,
        2_000_000_000,
        stage=chat._STAGE_R_SFT,
    ) == ("100m-2b-rsft-r0-16716-e3-001", chat._SOURCE_R_SFT)


def test_explicit_rsft_run_id_overrides_registered_default() -> None:
    run_id = "100m-2b-rsft-r0-atomic-repeat-e10-001"
    assert chat._resolve_chat_run(
        100_000_000,
        2_000_000_000,
        stage=chat._STAGE_R_SFT,
        run_id=run_id,
    ) == (run_id, chat._SOURCE_R_SFT)


def test_explicit_sft_run_id_overrides_registered_default() -> None:
    run_id = "100m-2b-sft-s0-10pct-001"
    assert chat._resolve_chat_run(
        100_000_000,
        2_000_000_000,
        stage=chat._STAGE_SFT,
        run_id=run_id,
    ) == (run_id, chat._SOURCE_SFT)


def test_run_id_override_rejects_pretrained_stage() -> None:
    with pytest.raises(RuntimeError, match="only with --sft or --r-sft"):
        chat._resolve_chat_run(
            100_000_000,
            2_000_000_000,
            stage=chat._STAGE_PRETRAINED,
            run_id="100m-2b-data-001",
        )


def test_parser_accepts_explicit_sft_run_id() -> None:
    args = chat._parse_args(
        [
            "--model_params",
            "100M",
            "--num_tokens",
            "2B",
            "--sft",
            "--run-id",
            "100m-2b-sft-s0-10pct-001",
        ]
    )
    assert args.stage == chat._STAGE_SFT
    assert args.run_id == "100m-2b-sft-s0-10pct-001"


def test_parser_accepts_repeat_rsft_run_id() -> None:
    args = chat._parse_args(
        [
            "--model_params",
            "100M",
            "--num_tokens",
            "2B",
            "--r-sft",
            "--run-id",
            "100m-2b-rsft-r0-atomic-repeat-e10-001",
        ]
    )
    assert args.stage == chat._STAGE_R_SFT
    assert args.run_id == "100m-2b-rsft-r0-atomic-repeat-e10-001"
