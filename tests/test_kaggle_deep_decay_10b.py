"""CPU-only contracts for the 100M/10B Kaggle dual-T4 continuation."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / "kaggle"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_deep_decay_schedule_and_exact_source_are_frozen() -> None:
    module = _load("small_llm_kaggle_deep_decay_test", KAGGLE / "deep_decay_10b_from_15500.py")
    assert module.SOURCE_CHECKPOINT_ID == "step-00015500"
    assert module.SOURCE_EXPECTED_TOKENS == 2_031_616_000
    assert module.SEQUENCES_PER_BLOCK == 64
    assert module.SOURCE_MICROBATCH_SIZE == 4
    assert module.MICROBATCH_SIZE == 2
    assert module.SETTLE_END_STEP == 17_789
    assert module.COOLDOWN_START_STEP == 73_242
    assert module.FINAL_STEP == 76_294
    assert module.TOTAL_TARGETS == 10_000_007_168
    assert math.isclose(module._expected_lr(module.SETTLE_END_TOKENS), 1e-4, rel_tol=1e-12)
    assert math.isclose(module._expected_lr(module.COOLDOWN_START_TOKENS), 1e-5, rel_tol=1e-12)
    assert math.isclose(module._expected_lr(module.TOTAL_TARGETS), 5e-6, rel_tol=1e-12)


def test_dual_t4_command_preserves_block64_and_t4_safe_microbatch() -> None:
    module = _load("small_llm_kaggle_deep_decay_command_test", KAGGLE / "deep_decay_10b_from_15500.py")
    trainer = [
        "/usr/bin/python",
        "-m",
        "trainer",
        "--sequences-per-block",
        "64",
        "--microbatch-size",
        "2",
        "--schedule",
        "wsqd",
    ]
    with mock.patch.object(module.shutil, "which", return_value="/usr/bin/uv"):
        command = module._dual_t4_command(trainer)
    assert "torch.distributed.run" in command
    assert "--nproc-per-node=2" in command
    assert str(KAGGLE / "dual_t4_train_block64.py") in command
    assert command[command.index("--sequences-per-block") + 1] == "64"
    assert command[command.index("--microbatch-size") + 1] == "2"
    assert "torch==2.10.0" in command
    assert "triton==3.6.0" in command
    assert "fla-core==0.5.2" in command


def test_block64_wrapper_reuses_shared_exact_batch_ddp() -> None:
    source = (KAGGLE / "dual_t4_train_block64.py").read_text(encoding="utf-8")
    assert "SEQUENCES_PER_BLOCK = 64" in source
    assert "MICROBATCH_SIZE = 2" in source
    assert "base.SEQUENCES_PER_BLOCK = SEQUENCES_PER_BLOCK" in source
    assert "base.MICROBATCH_SIZE = MICROBATCH_SIZE" in source
    assert "microbatch_size: int = MICROBATCH_SIZE" in source
    assert "return base.main(argv)" in source
    assert "32 sequences/rank" in source


def test_source_fork_changes_only_execution_slicing_plus_authorized_schedule() -> None:
    source = (KAGGLE / "deep_decay_10b_from_15500.py").read_text(encoding="utf-8")
    assert 'if config.get("microbatch_size") != SOURCE_MICROBATCH_SIZE' in source
    assert "microbatch_size=MICROBATCH_SIZE" in source
    assert 'schedule="wsqd"' in source
    assert 'schedule_anchor_tokens=SOURCE_EXPECTED_TOKENS' in source
    assert 'base_power=BASE_POWER' in source


def test_canonical_launcher_exposes_only_100m_10b_deep_decay_action() -> None:
    source = (KAGGLE / "launch.py").read_text(encoding="utf-8")
    assert '"deep-decay"' in source
    assert "DEEP_DECAY_MODEL = 100_000_000" in source
    assert "DEEP_DECAY_TOKENS = 10_000_000_000" in source
    assert "deep_decay_10b_from_15500.py" in source
    assert "deep-decay is frozen to --model 100M --tokens 10B" in source


def test_dry_run_describes_exact_global_optimizer_geometry() -> None:
    module = _load("small_llm_kaggle_deep_decay_dry_run_test", KAGGLE / "deep_decay_10b_from_15500.py")
    payload = module._dry_run_payload(250)
    assert payload["execution"] == "kaggle_dual_t4_ddp_block64"
    assert payload["world_size"] == 2
    assert payload["sequences_per_block"] == 64
    assert payload["sequences_per_rank"] == 32
    assert payload["source_microbatch_size"] == 4
    assert payload["microbatch_size"] == 2
    assert payload["local_microbatches_per_rank"] == 16
    assert payload["remote_checkpoint_every"] == 250
    assert payload["source_checkpoint_id"] == "step-00015500"
