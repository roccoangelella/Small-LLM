from __future__ import annotations

import argparse
from types import SimpleNamespace

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


def test_raw_prompt_is_exact_and_rejects_overlong_input() -> None:
    encoding = _ByteEncoding()
    assert chat._encode_raw_prompt(" France is ", encoding=encoding, max_prompt_tokens=11) == list(
        b" France is "
    )
    with pytest.raises(ValueError, match="maximum is 10"):
        chat._encode_raw_prompt(" France is ", encoding=encoding, max_prompt_tokens=10)
    with pytest.raises(ValueError, match="no tokens"):
        chat._encode_raw_prompt("", encoding=encoding, max_prompt_tokens=10)


def test_chat_uses_standalone_raw_prompts_and_unlabelled_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    prompts = iter([" The capital of France is ", "Once upon a time", "/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(prompts))
    monkeypatch.setattr(chat, "_build_chat_encoding", lambda **kwargs: _ByteEncoding())
    calls = []

    def fake_sample(model, prompt_ids, **kwargs):
        calls.append((prompt_ids, kwargs["seed"]))
        print(("Paris", "a fox")[len(calls) - 1], end="")
        return []

    monkeypatch.setattr(chat, "_stream_sample_token_ids", fake_sample)
    chat._chat(
        object(), SimpleNamespace(max_seq_len=256, semantic_vocab_size=50_257),
        device=SimpleNamespace(type="cpu"), stage=chat._STAGE_PRETRAINED,
    )
    assert calls == [
        (list(b" The capital of France is "), chat.SEED),
        (list(b"Once upon a time"), chat.SEED),
    ]
    output = capsys.readouterr().out
    assert "Paris\n" in output and "a fox\n" in output
    assert "User:" not in output and "Assistant:" not in output
    assert "you>" not in output and "assistant>" not in output


def test_every_prompt_is_independent_and_clear_is_not_a_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    prompts = iter(["first", "second", "first", "/clear", "/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(prompts))
    monkeypatch.setattr(chat, "_build_chat_encoding", lambda **kwargs: _ByteEncoding())
    calls = []

    def fake_sample(model, prompt_ids, **kwargs):
        calls.append((prompt_ids, kwargs["seed"]))
        print(f"output-{kwargs['seed']}", end="")
        return []

    monkeypatch.setattr(chat, "_stream_sample_token_ids", fake_sample)
    chat._chat(
        object(), SimpleNamespace(max_seq_len=256, semantic_vocab_size=50_257),
        device=SimpleNamespace(type="cpu"), stage=chat._STAGE_PRETRAINED,
    )
    assert calls == [
        (list(b"first"), chat.SEED),
        (list(b"second"), chat.SEED),
        (list(b"first"), chat.SEED),
        (list(b"/clear"), chat.SEED),
    ]
    output = capsys.readouterr().out
    assert "new chat started" not in output
    assert output.count(f"output-{chat.SEED}\n") == 4


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


def test_download_model_storage_bucket_fails_closed_on_missing_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    def fake_download_bucket(**kwargs):
        raise RuntimeError("bucket not found")

    def fake_download_artifact(**kwargs):
        pytest.fail("must not fall back to a different artifact source")

    monkeypatch.setattr(
        "trainer.post_pretraining_prompt_suite_bucket.download_verified_bucket_checkpoint",
        fake_download_bucket,
    )
    monkeypatch.setattr(
        "trainer.model_artifact.download_verified_model_artifact",
        fake_download_artifact,
    )
    monkeypatch.setattr(chat, "_CHAT_MODEL_CACHE_DIR", tmp_path / "chat_models")

    with pytest.raises(RuntimeError, match="bucket not found"):
        chat._download_model(
            repo_id="owner/repo",
            run_id=chat._DEFAULT_MOE_RUN_ID,
            source=chat._SOURCE_STORAGE_BUCKET,
            stage=chat._STAGE_PRETRAINED,
            device="cpu",
        )
    assert not (tmp_path / "chat_models" / chat._STAGE_PRETRAINED / chat._DEFAULT_MOE_RUN_ID).exists()


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
        "seed_policy": "fixed_seed_per_prompt",
        "max_seq_len": 1024,
        "eos_token_id": 50_256,
        "precision": "fp16",
    }


def test_moe_cli_resolution() -> None:
    args_moe = chat._parse_args(["--moe"])
    assert args_moe.moe
    assert args_moe.stage == chat._STAGE_PRETRAINED
    assert args_moe.model_params is None
    assert args_moe.num_tokens is None
    assert not args_moe.allow_incomplete
    assert chat._resolve_chat_run(
        stage=args_moe.stage,
        moe=args_moe.moe,
    ) == (chat._DEFAULT_MOE_RUN_ID, chat._SOURCE_STORAGE_BUCKET)

    args_moe_explicit = chat._parse_args(
        ["--moe", "--pre-trained", "--allow-incomplete"]
    )
    assert args_moe_explicit.moe
    assert args_moe_explicit.allow_incomplete
    assert args_moe_explicit.stage == chat._STAGE_PRETRAINED
    assert chat._resolve_chat_run(
        stage=args_moe_explicit.stage,
        moe=args_moe_explicit.moe,
    ) == (chat._DEFAULT_MOE_RUN_ID, chat._SOURCE_STORAGE_BUCKET)

    args_run_id = chat._parse_args(
        ["--run-id", "moe-100b-superbpe-001", "--pre-trained"]
    )
    assert chat._resolve_chat_run(
        stage=args_run_id.stage,
        run_id=args_run_id.run_id,
    ) == ("moe-100b-superbpe-001", chat._SOURCE_STORAGE_BUCKET)
    assert chat._parse_args(["--run-id", chat._DEFAULT_MOE_RUN_ID]).stage == chat._STAGE_PRETRAINED

    with pytest.raises(SystemExit):
        chat._parse_args(["--moe", "--run-id", "not-a-moe-run"])
    with pytest.raises(RuntimeError, match="only supports --pre-trained"):
        chat._resolve_chat_run(stage=chat._STAGE_SFT, run_id=chat._DEFAULT_MOE_RUN_ID)
    with pytest.raises(SystemExit):
        chat._parse_args(["--moe", "--sft"])

    # 100M 100B is not a registered profile
    with pytest.raises(RuntimeError, match="no registered pre-trained chat profile"):
        chat._resolve_chat_run(
            100_000_000,
            100_000_000_000,
            stage=chat._STAGE_PRETRAINED,
        )


def test_superbpe_eos_token_id_and_generation_settings() -> None:
    class MoEConfig:
        semantic_vocab_size = 8_000
        max_seq_len = 2048

    class StandardConfig:
        semantic_vocab_size = 50_257
        max_seq_len = 2048

    class Device:
        type = "cpu"

    assert chat._resolve_eos_token_id(MoEConfig()) == 7_992
    assert chat._resolve_eos_token_id(StandardConfig()) == 50_256

    settings = chat._generation_settings(
        MoEConfig(), device=Device(), run_id=chat._DEFAULT_MOE_RUN_ID
    )
    assert settings["eos_token_id"] == 7_992
    assert settings["precision"] == "fp32"


def test_superbpe_chat_encoding_encode_decode_stream() -> None:
    encoding = chat.SuperBPEChatEncoding()
    text = "Hello, world! 🍕 Ciao caffè."
    ids = encoding.encode(text)
    assert isinstance(ids, list)
    assert ids
    assert encoding.decode(ids) == text

    streamer = chat._TokenTextStreamer(encoding)
    chunks = [streamer.push(tid) for tid in ids]
    chunks.append(streamer.finish())
    assert "".join(chunks) == text


def test_build_chat_encoding_selects_pinned_superbpe_per_run() -> None:
    class MoEConfig:
        semantic_vocab_size = 8_000
        max_seq_len = 2048

    for run_id, expected_filename in (
        ("moe-100b-superbpe-001", "superbpe_8000.json"),
        (chat._DEFAULT_MOE_RUN_ID, "superbpe_8000_v2.json"),
    ):
        encoding = chat._build_chat_encoding(
            stage=chat._STAGE_PRETRAINED,
            config=MoEConfig(),
            run_id=run_id,
        )
        assert isinstance(encoding, chat.SuperBPEChatEncoding)
        settings = chat._generation_settings(MoEConfig(), device=type("Device", (), {"type": "cpu"})(), run_id=run_id)
        assert settings["tokenizer_artifact"] == f"tokenizer/{expected_filename}"
        assert settings["tokenizer_sha256"] == chat._MOE_CHAT_TOKENIZERS[run_id][1]
        assert encoding.decode(encoding.encode("The capital of France is Paris.")) == (
            "The capital of France is Paris."
        )

    previous = chat._build_chat_encoding(
        stage=chat._STAGE_PRETRAINED, config=MoEConfig(), run_id="moe-100b-superbpe-001"
    )
    latest = chat._build_chat_encoding(
        stage=chat._STAGE_PRETRAINED, config=MoEConfig(), run_id=chat._DEFAULT_MOE_RUN_ID
    )
    assert previous.encode("The capital of France is Paris.") != latest.encode(
        "The capital of France is Paris."
    )
    with pytest.raises(RuntimeError, match="no verified SuperBPE tokenizer identity"):
        chat._build_chat_encoding(
            stage=chat._STAGE_PRETRAINED, config=MoEConfig(), run_id="moe-unknown"
        )
    with pytest.raises(RuntimeError, match="no verified SuperBPE tokenizer identity"):
        chat._generation_settings(MoEConfig(), device=type("Device", (), {"type": "cpu"})())


def test_superbpe_artifact_drift_rejected(tmp_path) -> None:
    path = tmp_path / "modified.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact identity mismatch"):
        chat.SuperBPEChatEncoding(path, expected_sha256=chat._MOE_CHAT_TOKENIZERS[chat._DEFAULT_MOE_RUN_ID][1])


def test_load_completed_checkpoint_handles_moe_and_superbpe(tmp_path) -> None:
    import hashlib
    import json
    import pickle
    import torch
    from MOE_model.config import MoEModelConfig
    from MOE_model.model import MoESmallLLM
    from model.config import ModelConfig

    dense = ModelConfig(
        semantic_vocab_size=8000,
        padded_vocab_size=8192,
        max_seq_len=16,
        d_model=64,
        n_layers=4,
        d_ff=96,
        n_heads=2,
        head_dim=32,
        gdn_num_key_heads=2,
        gdn_num_value_heads=2,
        gdn_key_dim=32,
        gdn_value_dim=32,
        gdn_conv_kernel_size=4,
        gdn_chunk_size=4,
    )
    moe_config = MoEModelConfig(
        dense=dense,
        num_experts=8,
        top_k=1,
        expert_d_ff=96,
        moe_layer_indices=(0, 1, 2, 3),
    )
    model = MoESmallLLM(moe_config)

    checkpoint_dir = tmp_path / "moe_checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "checkpoint.json").write_text(
        json.dumps({"version": 1, "pipeline_state": {}}), encoding="utf-8"
    )

    trainer_state = {
        "version": 1,
        "config": {
            "schedule": "wsd",
            "warmup_tokens": 100,
            "stable_tokens": 800,
            "decay_tokens": 100,
        },
        "consumed_tokens": 500,
        "model_config": moe_config.as_dict(),
        "model": model.state_dict(),
    }
    with (checkpoint_dir / "trainer_state.pkl").open("wb") as handle:
        pickle.dump(trainer_state, handle)

    def _sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    manifest = {
        "version": 1,
        "files": [
            {
                "name": "checkpoint.json",
                "sha256": _sha(checkpoint_dir / "checkpoint.json"),
                "byte_size": (checkpoint_dir / "checkpoint.json").stat().st_size,
            },
            {
                "name": "trainer_state.pkl",
                "sha256": _sha(checkpoint_dir / "trainer_state.pkl"),
                "byte_size": (checkpoint_dir / "trainer_state.pkl").stat().st_size,
            },
        ],
    }
    (checkpoint_dir / "local_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    # Incomplete without permission raises
    with pytest.raises(RuntimeError, match="not complete"):
        chat._load_completed_checkpoint(
            checkpoint_dir,
            device=torch.device("cpu"),
            stage=chat._STAGE_PRETRAINED,
            run_id="other-run",
            allow_incomplete=False,
        )

    # Incomplete with allow_incomplete=True succeeds and returns MoESmallLLM
    loaded_model, loaded_config, consumed, _ = chat._load_completed_checkpoint(
        checkpoint_dir,
        device=torch.device("cpu"),
        stage=chat._STAGE_PRETRAINED,
        run_id="other-run",
        allow_incomplete=True,
    )
    assert isinstance(loaded_model, MoESmallLLM)
    assert loaded_config.semantic_vocab_size == 8000
    assert consumed == 500

    # The pinned, verified chat snapshot is an intermediate training checkpoint.
    moe_loaded, _, _, _ = chat._load_completed_checkpoint(
        checkpoint_dir,
        device=torch.device("cpu"),
        stage=chat._STAGE_PRETRAINED,
        run_id=chat._DEFAULT_MOE_RUN_ID,
        allow_incomplete=False,
    )
    assert isinstance(moe_loaded, MoESmallLLM)
