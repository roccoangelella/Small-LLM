from __future__ import annotations

import argparse

import pytest

import chat


def test_parse_quantity_accepts_profile_spellings() -> None:
    assert chat._parse_quantity("20M") == 20_000_000
    assert chat._parse_quantity("100M") == 100_000_000
    assert chat._parse_quantity("500m") == 500_000_000
    assert chat._parse_quantity("2B") == 2_000_000_000
    assert chat._parse_quantity("10B") == 10_000_000_000
    assert chat._parse_quantity("20_000_000") == 20_000_000


def test_parse_quantity_rejects_non_integral_sizes() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        chat._parse_quantity("0")
    with pytest.raises(argparse.ArgumentTypeError):
        chat._parse_quantity("1.5")
    with pytest.raises(argparse.ArgumentTypeError):
        chat._parse_quantity("20Q")


def test_resolve_chat_run_is_stage_explicit_and_fail_closed() -> None:
    assert chat._resolve_chat_run(
        100_000_000,
        2_000_000_000,
        stage=chat._STAGE_PRETRAINED,
    ) == (
        "100m-2b-data-001",
        chat._SOURCE_STABLE_MODEL,
    )
    assert chat._resolve_chat_run(
        20_000_000,
        500_000_000,
        stage=chat._STAGE_SFT,
    ) == (
        "20m-500m-sft-s0-001",
        chat._SOURCE_SFT,
    )
    assert chat._resolve_chat_run(
        100_000_000,
        2_000_000_000,
        stage=chat._STAGE_SFT,
    ) == (
        "100m-2b-sft-s0-10pct-peak3000-001",
        chat._SOURCE_SFT,
    )
    assert chat._resolve_chat_run(
        100_000_000,
        10_000_000_000,
        stage=chat._STAGE_SFT,
    ) == (
        "100m-10b-sft-s0-2b10pct-data-001",
        chat._SOURCE_SFT,
    )
    assert chat._resolve_chat_run(
        100_000_000,
        2_000_000_000,
        stage=chat._STAGE_R_SFT,
    ) == (
        "100m-2b-rsft-r0-16716-e3-001",
        chat._SOURCE_R_SFT,
    )
    assert chat._resolve_chat_run(
        100_000_000,
        10_000_000_000,
        stage=chat._STAGE_PRETRAINED,
    ) == (
        "100m-10b-deep-decay-from-step15500",
        chat._SOURCE_STORAGE_BUCKET,
    )
    with pytest.raises(RuntimeError, match="no registered pre-trained chat profile"):
        chat._resolve_chat_run(
            20_000_000,
            10_000_000_000,
            stage=chat._STAGE_PRETRAINED,
        )
    with pytest.raises(RuntimeError, match="no registered r-sft chat profile"):
        chat._resolve_chat_run(
            20_000_000,
            2_000_000_000,
            stage=chat._STAGE_R_SFT,
        )


def test_storage_bucket_repo_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SMALL_LLM_HF_REPO_ID", "owner/base")
    monkeypatch.delenv("SMALL_LLM_HF_CHECKPOINT_BUCKET_ID", raising=False)
    assert chat._repo_id(source=chat._SOURCE_STORAGE_BUCKET) == "owner/base"

    monkeypatch.setenv("SMALL_LLM_HF_CHECKPOINT_BUCKET_ID", "owner/explicit-bucket")
    monkeypatch.delenv("SMALL_LLM_HF_REPO_ID", raising=False)
    assert chat._repo_id(source=chat._SOURCE_STORAGE_BUCKET) == "owner/explicit-bucket"

    monkeypatch.delenv("SMALL_LLM_HF_CHECKPOINT_BUCKET_ID", raising=False)
    with pytest.raises(RuntimeError, match="set SMALL_LLM_HF_REPO_ID"):
        chat._repo_id(source=chat._SOURCE_STORAGE_BUCKET)


def test_100m_10b_sft_cli_selects_completed_registered_run() -> None:
    args = chat._parse_args(
        ["--model_params", "100M", "--num_tokens", "10B", "--sft"]
    )
    assert args.model_params == 100_000_000
    assert args.num_tokens == 10_000_000_000
    assert args.stage == chat._STAGE_SFT
    assert chat._resolve_chat_run(
        args.model_params,
        args.num_tokens,
        stage=args.stage,
    ) == (
        "100m-10b-sft-s0-2b10pct-data-001",
        chat._SOURCE_SFT,
    )


def test_rsft_repo_resolution_prefers_dedicated_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SMALL_LLM_HF_REPO_ID", "owner/base")
    monkeypatch.setenv("SMALL_LLM_SFT_HF_REPO_ID", "owner/sft")
    monkeypatch.setenv("SMALL_LLM_RSFT_HF_REPO_ID", "owner/rsft")
    assert chat._repo_id(source=chat._SOURCE_R_SFT) == "owner/rsft"

    monkeypatch.delenv("SMALL_LLM_RSFT_HF_REPO_ID")
    assert chat._repo_id(source=chat._SOURCE_R_SFT) == "owner/sft"


def test_chat_stage_flag_is_mandatory_and_mutually_exclusive() -> None:
    base = ["--model_params", "100M", "--num_tokens", "2B"]
    with pytest.raises(SystemExit):
        chat._parse_args(base)
    with pytest.raises(SystemExit):
        chat._parse_args([*base, "--sft", "--r-sft"])

    pretrained = chat._parse_args([*base, "--pre-trained"])
    sft = chat._parse_args([*base, "--sft"])
    rsft = chat._parse_args([*base, "--r-sft"])
    assert pretrained.stage == chat._STAGE_PRETRAINED
    assert sft.stage == chat._STAGE_SFT
    assert rsft.stage == chat._STAGE_R_SFT

    pretrained_10b = chat._parse_args(
        ["--model_params", "100M", "--num_tokens", "10B", "--pre-trained"]
    )
    assert pretrained_10b.stage == chat._STAGE_PRETRAINED
    assert pretrained_10b.model_params == 100_000_000
    assert pretrained_10b.num_tokens == 10_000_000_000


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


class _ByteEncoding:
    def encode(self, text: str, **kwargs) -> list[int]:
        del kwargs
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int]) -> str:
        return bytes(token_ids).decode("utf-8")

    def decode_single_token_bytes(self, token_id: int) -> bytes:
        return bytes([token_id])


def test_chat_tokenizer_selection_keeps_normal_stages_plain_and_rsft_extended() -> None:
    base = _ByteEncoding()
    assert chat._build_chat_encoding(
        stage=chat._STAGE_PRETRAINED,
        base_encoding=base,
    ) is base
    assert chat._build_chat_encoding(
        stage=chat._STAGE_SFT,
        base_encoding=base,
    ) is base

    tokenizer = chat._load_rsft_tokenizer_module()
    spec = tokenizer.ReasoningTokenSpec(
        reasoning_start="<R>",
        reasoning_end="</R>",
        answer_start="<A>",
    )
    extended = chat._build_chat_encoding(
        stage=chat._STAGE_R_SFT,
        reasoning_spec=spec,
        base_encoding=base,
    )
    assert extended.encode("<R>x</R><A>y") == [50_257, ord("x"), 50_258, 50_259, ord("y")]
    assert extended.decode([50_257, ord("x"), 50_258, 50_259, ord("y")]) == "<R>x</R><A>y"


def test_rsft_chat_requires_reasoning_token_specification() -> None:
    with pytest.raises(RuntimeError, match="requires a verified reasoning token specification"):
        chat._build_chat_encoding(
            stage=chat._STAGE_R_SFT,
            base_encoding=_ByteEncoding(),
        )


def test_accepted_rsft_protocol_is_atomic_and_canonical() -> None:
    assert chat._R_SFT_CANONICAL_MARKERS == ("<think>", "</think>", "<answer>")


def test_download_model_storage_bucket_persists_and_reuses_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    recorded = {"bucket_calls": 0}

    def fake_download_bucket(*, repo_id, run_id, token, revision, pointer_name, destination):
        recorded["bucket_calls"] += 1
        recorded["bucket"] = {
            "repo_id": repo_id,
            "run_id": run_id,
            "token": token,
            "revision": revision,
            "pointer_name": pointer_name,
            "destination": destination,
        }
        checkpoint_root = destination / "step-00076294"
        checkpoint_root.mkdir(parents=True)
        return checkpoint_root, {"checkpoint_id": "step-00076294"}

    loaded_roots = []

    def fake_load(checkpoint_root, *, device, stage):
        loaded_roots.append(checkpoint_root)
        return "model", "config", 10_000_007_168, None

    monkeypatch.setattr(
        "trainer.post_pretraining_prompt_suite_bucket.download_verified_bucket_checkpoint",
        fake_download_bucket,
    )
    monkeypatch.setattr(chat, "_load_completed_checkpoint", fake_load)
    monkeypatch.setattr(chat, "_CHAT_MODEL_CACHE_DIR", tmp_path / "chat_models")
    monkeypatch.setenv("HF_TOKEN", "test-token")

    first = chat._download_model(
        repo_id="owner/repo",
        run_id="100m-10b-deep-decay-from-step15500",
        source=chat._SOURCE_STORAGE_BUCKET,
        stage=chat._STAGE_PRETRAINED,
        device="cpu",
    )
    model, config, consumed, reasoning_spec, info, cache_root = first
    assert model == "model"
    assert config == "config"
    assert consumed == 10_000_007_168
    assert reasoning_spec is None
    assert info["checkpoint_id"] == "step-00076294"
    assert info["cache_status"] == "downloaded"
    assert recorded["bucket"]["run_id"] == "100m-10b-deep-decay-from-step15500"
    assert recorded["bucket"]["pointer_name"] == "latest"
    assert recorded["bucket"]["token"] == "test-token"
    assert cache_root == (
        tmp_path
        / "chat_models"
        / chat._STAGE_PRETRAINED
        / "100m-10b-deep-decay-from-step15500"
    )
    assert (cache_root / chat._CHAT_MODEL_CACHE_METADATA).is_file()

    second = chat._download_model(
        repo_id="owner/repo",
        run_id="100m-10b-deep-decay-from-step15500",
        source=chat._SOURCE_STORAGE_BUCKET,
        stage=chat._STAGE_PRETRAINED,
        device="cpu",
    )
    assert second[4]["cache_status"] == "hit"
    assert second[5] == cache_root
    assert recorded["bucket_calls"] == 1
    assert loaded_roots == [
        cache_root / "step-00076294",
        cache_root / "step-00076294",
    ]


def test_download_model_storage_bucket_falls_back_to_model_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    def fake_download_bucket(**kwargs):
        raise RuntimeError("bucket not found")

    recorded = {}

    def fake_download_artifact(*, repo_id, run_id, token, revision, destination):
        recorded["artifact"] = {
            "repo_id": repo_id,
            "run_id": run_id,
            "token": token,
            "revision": revision,
            "destination": destination,
        }
        checkpoint_root = destination / "step-00076294"
        checkpoint_root.mkdir(parents=True)
        return checkpoint_root, {"checkpoint_id": "step-00076294"}

    def fake_load(checkpoint_root, *, device, stage):
        return "model", "config", 10_000_007_168, None

    monkeypatch.setattr(
        "trainer.post_pretraining_prompt_suite_bucket.download_verified_bucket_checkpoint",
        fake_download_bucket,
    )
    monkeypatch.setattr(
        "trainer.model_artifact.download_verified_model_artifact",
        fake_download_artifact,
    )
    monkeypatch.setattr(chat, "_load_completed_checkpoint", fake_load)
    monkeypatch.setattr(chat, "_CHAT_MODEL_CACHE_DIR", tmp_path / "chat_models")

    _model, _config, _consumed, _reasoning_spec, info, cache_root = chat._download_model(
        repo_id="owner/repo",
        run_id="100m-10b-deep-decay-from-step15500",
        source=chat._SOURCE_STORAGE_BUCKET,
        stage=chat._STAGE_PRETRAINED,
        device="cpu",
    )
    assert recorded["artifact"]["run_id"] == "100m-10b-deep-decay-from-step15500"
    assert info["checkpoint_id"] == "step-00076294"
    assert info["cache_status"] == "downloaded"
    assert cache_root.is_dir()


def test_generation_settings_report_effective_chat_sampler() -> None:
    class Config:
        max_seq_len = 1024

    class Device:
        type = "cuda"

    settings = chat._generation_settings(Config(), device=Device())
    assert settings == {
        "temperature": chat.TEMPERATURE,
        "top_p": chat.TOP_P,
        "top_k": chat.TOP_K,
        "max_new_tokens": chat.MAX_NEW_TOKENS,
        "base_seed": chat.SEED,
        "seed_policy": "base_seed + zero_based_turn_index",
        "max_seq_len": 1024,
        "eos_token_id": 50_256,
        "precision": "fp16",
    }
