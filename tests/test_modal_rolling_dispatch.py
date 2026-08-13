"""Source-level guardrails for the Modal CPU-before-H100 rolling dataset contract."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_incremental_producer_and_supervised_stage_precede_h100_spawn() -> None:
    source = (ROOT / "modal" / "launch.py").read_text(encoding="utf-8")

    producer = source.index("producer_call = produce_rolling_dataset_remote.spawn")
    stage = source.index("stage_call = stage_rolling_dataset_remote.spawn")
    supervise = source.index("await_stage_with_producer(stage_call, producer_call)")
    authorize = source.index('staged.get("h100_dispatch_allowed") is not True')
    spawn = source.index("result = train_rolling_remote.with_options")
    assert producer < stage < supervise < authorize < spawn
    assert '"h100_allocated": False' in source
    assert "stage_rolling_dataset_remote.remote" not in source


def test_dataset_producer_is_cpu_only_and_h100_has_no_automatic_retry() -> None:
    source = (ROOT / "modal" / "launch.py").read_text(encoding="utf-8")

    producer_decorator_start = source.rindex(
        "@app.function(", 0, source.index("def produce_rolling_dataset_remote")
    )
    producer_function_start = source.index("def produce_rolling_dataset_remote")
    producer_decorator = source[producer_decorator_start:producer_function_start]
    assert "cpu=4.0" in producer_decorator
    assert "memory=8192" in producer_decorator
    assert "gpu=" not in producer_decorator

    h100_decorator_start = source.rindex("@app.function(", 0, source.index("def train_rolling_remote"))
    h100_function_start = source.index("def train_rolling_remote")
    h100_decorator = source[h100_decorator_start:h100_function_start]
    assert "gpu=DEFAULT_GPU" in h100_decorator
    assert "retries=" not in h100_decorator


def test_cpu_stage_uses_checkpoint_aligned_incremental_lead_window() -> None:
    source = (ROOT / "modal" / "rolling_dataset.py").read_text(encoding="utf-8")

    assert "next_unconsumed_block" in source
    assert "stage_incremental_window" in source
    assert 'start_block_id=int(cursor["next_block_id"])' in source
    assert "checkpoint advanced after CPU dataset staging" in source
    assert "verify_incremental_stage" in source


def test_modal_microbatch_probe_does_not_start_one_gib_successor_prefetch() -> None:
    source = (ROOT / "trainer" / "cli_setup.py").read_text(encoding="utf-8")

    assert 'os.environ.get("SMALL_LLM_MODAL_ROLLING_DATASET") == "1"' in source
    assert 'getattr(args, "wandb_mode", "disabled") == "disabled"' in source
    assert "return None" in source
