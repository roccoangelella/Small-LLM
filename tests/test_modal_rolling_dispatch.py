"""Source-level guardrails for the Modal CPU-before-H100 rolling dataset contract."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_rolling_launcher_stages_and_authorizes_before_h100_spawn() -> None:
    source = (ROOT / "modal" / "launch.py").read_text(encoding="utf-8")

    stage = source.index("staged = stage_rolling_dataset_remote.remote")
    authorize = source.index('staged.get("h100_dispatch_allowed") is not True')
    spawn = source.index("result = train_rolling_remote.with_options")
    assert stage < authorize < spawn
    assert '"h100_allocated": False' in source


def test_rolling_h100_function_has_no_automatic_retry_bypassing_cpu_restage() -> None:
    source = (ROOT / "modal" / "launch.py").read_text(encoding="utf-8")
    decorator_start = source.rindex("@app.function(", 0, source.index("def train_rolling_remote"))
    function_start = source.index("def train_rolling_remote")
    decorator = source[decorator_start:function_start]

    assert "gpu=DEFAULT_GPU" in decorator
    assert "retries=" not in decorator
    assert "Deliberately no automatic function retry" in source


def test_cpu_stage_uses_checkpoint_aligned_next_block_not_always_shard_zero() -> None:
    source = (ROOT / "modal" / "rolling_dataset.py").read_text(encoding="utf-8")

    assert "next_unconsumed_block" in source
    assert 'start_block_id=int(cursor["next_block_id"])' in source
    assert "checkpoint advanced after CPU dataset staging" in source


def test_modal_microbatch_probe_does_not_start_one_gib_successor_prefetch() -> None:
    source = (ROOT / "trainer" / "cli_setup.py").read_text(encoding="utf-8")

    assert 'os.environ.get("SMALL_LLM_MODAL_ROLLING_DATASET") == "1"' in source
    assert 'getattr(args, "wandb_mode", "disabled") == "disabled"' in source
    assert "return None" in source
