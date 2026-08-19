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


def _token_metadata() -> dict[str, object]:
    return {
        "version": 1,
        "base_encoding": "gpt2",
        "semantic_vocab_size": 50_260,
        "special_tokens": {
            "reasoning_start": {"id": 50_257, "text": "<think>"},
            "reasoning_end": {"id": 50_258, "text": "</think>"},
            "answer_start": {"id": 50_259, "text": "<answer>"},
        },
    }


def _bundle(
    tmp_path: Path,
    *,
    targets: int = 63_000,
    delimiter: str = "atomic",
    contract: str | None = "atomic-production-v1",
    optimizer_target_tokens: int = 32_768,
) -> Path:
    root = tmp_path / f"bundle-{delimiter}-{contract or 'pilot'}"
    root.mkdir()
    rsft = {
        "stage": "r_sft_r0",
        "delimiter_format": delimiter,
        "reasoning_tokenizer": _token_metadata(),
        "reasoning_share_requested": 0.9,
        "retention_share_requested": 0.1,
    }
    if contract is not None:
        rsft["contract"] = contract
        rsft["reasoning_corpus_contract"] = "heterogeneous-groups-v1"
    (root / "bundle-manifest.json").write_text(
        json.dumps(
            {
                "schema": "small-llm-sft-bundle",
                "train_target_tokens_requested": targets,
                "optimizer_target_tokens": optimizer_target_tokens,
                "prepared_source": {
                    "dataset_name": "small-llm-rsft-r0-superior-instruction"
                },
                "rsft": rsft,
            }
        ),
        encoding="utf-8",
    )
    (root / "reasoning-tokens.json").write_text(
        json.dumps(_token_metadata()),
        encoding="utf-8",
    )
    return root


def _compact_token_spec(tmp_path: Path) -> Path:
    path = tmp_path / "reasoning-tokens-compact.json"
    path.write_text(
        json.dumps(
            {
                "reasoning_start": "<think>",
                "reasoning_end": "</think>",
                "answer_start": "<answer>",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_profile_is_fixed_to_100m_2b() -> None:
    profile = rsft_runtime.resolve_profile(
        100_000_000,
        2_000_000_000,
        run_id=rsft_runtime.PRODUCTION_RUN_ID,
        delimiter_format="atomic",
    )
    assert profile.parent_run_id == "100m-2b-sft-s0-001"
    assert profile.microbatch_size == 2
    assert profile.sft_run_id == "100m-2b-rsft-r0-001"

    with pytest.raises(rsft_runtime.base.RuntimeFailure):
        rsft_runtime.resolve_profile(
            20_000_000,
            2_000_000_000,
            run_id="unsupported",
            delimiter_format="atomic",
        )


def test_repeat_experiment_gets_epoch_specific_run_id() -> None:
    assert rsft_runtime.default_experiment_run_id("atomic", num_epochs=1) == (
        "100m-2b-rsft-r0-atomic-pilot-001"
    )
    assert rsft_runtime.default_experiment_run_id("atomic", num_epochs=10) == (
        "100m-2b-rsft-r0-atomic-repeat-e10-001"
    )


def test_bundle_budget_is_exact_not_parent_fraction(tmp_path: Path) -> None:
    adapter = _load_dual_adapter()
    bundle = _bundle(tmp_path, targets=77_777)
    assert adapter.bundle_target_budget(bundle) == 77_777


def test_compact_reasoning_token_spec_uses_fixed_promoted_ids(tmp_path: Path) -> None:
    adapter = _load_dual_adapter()
    tokenizer = _load_rsft_module("tokenizer")
    spec = adapter.load_reasoning_token_spec(_compact_token_spec(tmp_path), tokenizer)
    assert spec.special_tokens == {
        "<think>": 50_257,
        "</think>": 50_258,
        "<answer>": 50_259,
    }


def test_pipeline_identity_keeps_historical_ablation_distinct(tmp_path: Path) -> None:
    adapter = _load_dual_adapter()
    tokenizer = _load_rsft_module("tokenizer")
    token_spec = adapter.load_reasoning_token_spec(_compact_token_spec(tmp_path), tokenizer)
    common = {
        "parent_identity": {"identity_sha256": "a" * 64},
        "bundle_manifest": {"manifest_sha256": "b" * 64},
        "token_metadata": token_spec.to_metadata(),
    }
    atomic = adapter.rsft_pipeline_identity(delimiter_format="atomic", **common)
    textual = adapter.rsft_pipeline_identity(delimiter_format="textual", **common)
    repeated = adapter.rsft_pipeline_identity(
        delimiter_format="atomic",
        num_epochs=10,
        **common,
    )
    assert atomic["stage"] == "r_sft_r0"
    assert atomic["template_identity"] != textual["template_identity"]
    assert atomic["reasoning_tokenizer"] == textual["reasoning_tokenizer"]
    assert "num_epochs" not in atomic
    assert repeated["num_epochs"] == 10
    assert repeated["repeat_identity"] == "exact-block-replay-v1"
    assert repeated != atomic


def test_production_dry_run_is_atomic_only_and_uses_canonical_run_id(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle = _bundle(tmp_path, targets=63_000)
    result = rsft_cli.main(
        [
            "train",
            "--model",
            "100M",
            "--tokens",
            "2B",
            "--dataset-dir",
            str(bundle),
            "--parent-repo-id",
            "owner/parent",
            "--checkpoint-repo-id",
            "owner/checkpoints",
            "--dry-run",
        ]
    )
    assert result == 0
    output = capsys.readouterr().out
    assert '"contract": "atomic-production-v1"' in output
    assert '"delimiter_format": "atomic"' in output
    assert '"run_id": "100m-2b-rsft-r0-001"' in output
    assert '"bundle_target_tokens_one_pass": 63000' in output
    assert '"num_epochs": 1' in output
    assert "torch.distributed.run" in output
    assert "--nproc-per-node=2" in output
    assert "dual_t4_rsft.py" in output
    assert "--rsft-num-epochs" in output


def test_production_dry_run_auto_prepares_committed_superior_corpus(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = rsft_cli.main(
        [
            "train",
            "--model",
            "100M",
            "--tokens",
            "2B",
            "--parent-repo-id",
            "owner/parent",
            "--checkpoint-repo-id",
            "owner/checkpoints",
            "--dry-run",
        ]
    )
    assert result == 0
    output = capsys.readouterr().out
    assert "artifacts/rsft-superior-instruction-r0/reasoning.jsonl" in output
    assert '"reasoning_share": 0.9' in output
    assert '"s0_retention_share": 0.1' in output
    assert '"heldout_fraction_per_split": 0.01' in output
    assert '"bundle_target_tokens_one_pass": null' in output


def test_production_rejects_textual_bundle(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, delimiter="textual")
    with pytest.raises(SystemExit):
        rsft_cli.main(
            [
                "train",
                "--model",
                "100M",
                "--tokens",
                "2B",
                "--dataset-dir",
                str(bundle),
                "--parent-repo-id",
                "owner/parent",
                "--checkpoint-repo-id",
                "owner/checkpoints",
                "--dry-run",
            ]
        )


def test_production_rejects_pilot_optimizer_geometry(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, optimizer_target_tokens=2_048)
    with pytest.raises(SystemExit):
        rsft_cli.main(
            [
                "train",
                "--model",
                "100M",
                "--tokens",
                "2B",
                "--dataset-dir",
                str(bundle),
                "--parent-repo-id",
                "owner/parent",
                "--checkpoint-repo-id",
                "owner/checkpoints",
                "--dry-run",
            ]
        )


def test_production_rejects_repeat_epochs(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    with pytest.raises(SystemExit):
        rsft_cli.main(
            [
                "train",
                "--model",
                "100M",
                "--tokens",
                "2B",
                "--dataset-dir",
                str(bundle),
                "--num-epochs",
                "10",
                "--parent-repo-id",
                "owner/parent",
                "--checkpoint-repo-id",
                "owner/checkpoints",
                "--dry-run",
            ]
        )


def test_historical_ablation_still_has_explicit_textual_arm(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = rsft_cli.main(
        [
            "ablation",
            "--model",
            "100M",
            "--tokens",
            "2B",
            "--delimiter-format",
            "textual",
            "--parent-repo-id",
            "owner/parent",
            "--checkpoint-repo-id",
            "owner/checkpoints",
            "--dry-run",
        ]
    )
    assert result == 0
    output = capsys.readouterr().out
    assert '"contract": "pilot-ablation-v1"' in output
    assert '"delimiter_format": "textual"' in output
    assert '"run_id": "100m-2b-rsft-r0-textual-pilot-001"' in output
    assert '"num_epochs": 1' in output


def test_atomic_repeat_dry_run_is_10_exact_passes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = rsft_cli.main(
        [
            "ablation",
            "--model",
            "100M",
            "--tokens",
            "2B",
            "--delimiter-format",
            "atomic",
            "--num-epochs",
            "10",
            "--parent-repo-id",
            "owner/parent",
            "--checkpoint-repo-id",
            "owner/checkpoints",
            "--dry-run",
        ]
    )
    assert result == 0
    output = capsys.readouterr().out
    assert '"contract": "pilot-repeat-v1"' in output
    assert '"delimiter_format": "atomic"' in output
    assert '"run_id": "100m-2b-rsft-r0-atomic-repeat-e10-001"' in output
    assert '"num_epochs": 10' in output
    assert '"budget_mode": "bundle-exact-repeat"' in output
    assert "--rsft-num-epochs" in output
    assert "100m-2b-rsft-r0-atomic-repeat-e10-001" in output
