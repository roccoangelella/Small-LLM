from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
KAGGLE = REPO / "kaggle"
if str(KAGGLE) not in sys.path:
    sys.path.insert(0, str(KAGGLE))

import rsft_cli
import rsft_runtime


def _load_rsft_module(name: str):
    module_name = f"test_rsft_{name}"
    path = REPO / "post_training" / "R-SFT" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_dual_adapter():
    module_name = "test_dual_t4_rsft"
    path = KAGGLE / "dual_t4_rsft.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _bundle(tmp_path: Path, targets: int = 12_345) -> Path:
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "bundle-manifest.json").write_text(
        json.dumps({"train_target_tokens_requested": targets}),
        encoding="utf-8",
    )
    return root


def _token_spec(tmp_path: Path) -> Path:
    path = tmp_path / "reasoning-tokens.json"
    path.write_text(
        json.dumps(
            {
                "reasoning_start": "<|reasoning|>",
                "reasoning_end": "<|end_reasoning|>",
                "answer_start": "<|answer|>",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_profile_is_fixed_to_100m_2b() -> None:
    profile = rsft_runtime.resolve_profile(
        100_000_000,
        2_000_000_000,
        run_id="100m-2b-rsft-r0-test",
        delimiter_format="atomic",
    )
    assert profile.parent_run_id == "100m-2b-sft-s0-001"
    assert profile.microbatch_size == 2

    with pytest.raises(rsft_runtime.base.RuntimeFailure):
        rsft_runtime.resolve_profile(
            20_000_000,
            2_000_000_000,
            run_id="unsupported",
            delimiter_format="atomic",
        )


def test_bundle_budget_is_exact_not_parent_fraction(tmp_path: Path) -> None:
    adapter = _load_dual_adapter()
    bundle = _bundle(tmp_path, targets=77_777)
    assert adapter.bundle_target_budget(bundle) == 77_777


def test_compact_reasoning_token_spec_uses_fixed_promoted_ids(tmp_path: Path) -> None:
    adapter = _load_dual_adapter()
    tokenizer = _load_rsft_module("tokenizer")
    spec = adapter.load_reasoning_token_spec(_token_spec(tmp_path), tokenizer)
    assert spec.special_tokens == {
        "<|reasoning|>": 50_257,
        "<|end_reasoning|>": 50_258,
        "<|answer|>": 50_259,
    }


def test_pipeline_identity_separates_atomic_and_textual_arms(tmp_path: Path) -> None:
    adapter = _load_dual_adapter()
    tokenizer = _load_rsft_module("tokenizer")
    token_spec = adapter.load_reasoning_token_spec(_token_spec(tmp_path), tokenizer)
    common = {
        "parent_identity": {"identity_sha256": "a" * 64},
        "bundle_manifest": {"manifest_sha256": "b" * 64},
        "token_metadata": token_spec.to_metadata(),
    }
    atomic = adapter.rsft_pipeline_identity(delimiter_format="atomic", **common)
    textual = adapter.rsft_pipeline_identity(delimiter_format="textual", **common)
    assert atomic["stage"] == "r_sft_r0"
    assert atomic["template_identity"] != textual["template_identity"]
    assert atomic["reasoning_tokenizer"] == textual["reasoning_tokenizer"]


def test_dry_run_builds_two_t4_torchrun_command(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle = _bundle(tmp_path, targets=63_000)
    token_spec = _token_spec(tmp_path)
    result = rsft_cli.main(
        [
            "train",
            "--model",
            "100M",
            "--tokens",
            "2B",
            "--dataset-dir",
            str(bundle),
            "--run-id",
            "100m-2b-rsft-r0-atomic-pilot-test",
            "--delimiter-format",
            "atomic",
            "--token-spec",
            str(token_spec),
            "--parent-repo-id",
            "owner/parent",
            "--checkpoint-repo-id",
            "owner/checkpoints",
            "--dry-run",
        ]
    )
    assert result == 0
    output = capsys.readouterr().out
    assert '"topology": "2xTesla-T4-DDP"' in output
    assert '"bundle_target_tokens": 63000' in output
    assert "torch.distributed.run" in output
    assert "--nproc-per-node=2" in output
    assert "dual_t4_rsft.py" in output
    assert "--rsft-delimiter-format" in output
    assert "atomic" in output
