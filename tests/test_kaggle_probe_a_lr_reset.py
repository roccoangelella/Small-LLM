"Static contracts for the disposable Kaggle Probe A LR-reset launcher."
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "kaggle" / "probe_a_lr_reset_10b.py"
IMPL = ROOT / "kaggle" / "probe_a_lr_reset_10b_impl.py"


def test_probe_a_entrypoint_compiles() -> None:
    compile(ENTRYPOINT.read_text(encoding="utf-8"), str(ENTRYPOINT), "exec")


def test_probe_a_impl_compiles() -> None:
    compile(IMPL.read_text(encoding="utf-8"), str(IMPL), "exec")


def test_probe_a_entrypoint_reexecs_itself_for_hf_runtime() -> None:
    text = ENTRYPOINT.read_text(encoding="utf-8")
    assert "probe_a_lr_reset_10b_impl" in text
    assert "_ensure_probe_hf_bucket_runtime" in text
    assert "[kaggle-probe-a]" in text
    assert "str(Path(__file__).resolve())" in text
    assert "deep_decay._ensure_host_hf_bucket_runtime = _noop_hf_runtime_restart" in text


def test_probe_a_entrypoint_forces_100m_hf_identity() -> None:
    text = ENTRYPOINT.read_text(encoding="utf-8")
    assert "PROBE_BASE_HF_REPO_ID" in text
    assert "roccoangelella/small-llm-100m-qualification" in text
    assert "_force_probe_hf_identity" in text
    assert '"SMALL_LLM_HF_REPO_ID": PROBE_BASE_HF_REPO_ID' in text
    assert '"SMALL_LLM_HF_CHECKPOINT_BUCKET_ID"' in text
    assert '"SMALL_LLM_HF_DATASET_BUCKET_ID"' in text
    assert "probe_a_hf_identity_override" in text
    assert "_force_probe_hf_identity()" in text


def test_probe_a_entrypoint_pins_fixed_best_source_step() -> None:
    text = ENTRYPOINT.read_text(encoding="utf-8")
    assert 'PROBE_SOURCE_CHECKPOINT_ID = "step-00071750"' in text
    assert 'PROBE_SOURCE_KIND = "fixed_best_model_checkpoint"' in text
    assert "_restore_fixed_best_checkpoint" in text
    assert "best_model.json" in text
    assert "snapshot_download" in text
    assert "models/{impl._impl.RUN_ID}/{PROBE_SOURCE_CHECKPOINT_ID}" in text
    assert "_patch_impl_for_fixed_source" in text
    assert "fixed dedicated best-model checkpoint restore" in text
    assert "base_hf_repo_id" in text


def test_probe_a_reconstructs_missing_best_checkpoint_local_manifest() -> None:
    text = ENTRYPOINT.read_text(encoding="utf-8")
    assert "_ensure_best_source_local_manifest" in text
    assert "_POST_SAVE_METADATA" in text
    assert "trainer_state.pkl" in text
    assert "checkpoint.json" in text
    assert "sha256_path" in text
    assert "probe_a_rebuilt_local_manifest" in text
    assert "local_manifest_rebuilt" in text
    assert "verify_local_manifest(source)" in text
    assert "verify_local_manifest(staging)" in text


def test_probe_a_entrypoint_allows_new_fixed_step_wandb_runs() -> None:
    text = ENTRYPOINT.read_text(encoding="utf-8")
    assert "WANDB_RESUME=allow" in text
    assert 'impl._replace_option(command, "--wandb-resume", "allow")' in text
    assert "original_assert" in text
    assert "must use --wandb-resume allow" in text


def test_probe_a_impl_has_two_distinct_wandb_only_branches() -> None:
    text = IMPL.read_text(encoding="utf-8")
    assert "reset-low" in text
    assert "reset-mid" in text
    assert "100m-10b-probe-a-{branch.slug}-from-step{source_step}" in text
    assert "wandb_run_id_template" in text
    assert "--wandb-run-id" in text
    assert "--wandb-run-name" in text
    assert "--wandb-dir" in text
    assert "--wandb-mode" in text
    assert '"online"' in text


def test_probe_a_impl_isolates_wandb_identity_per_branch() -> None:
    text = IMPL.read_text(encoding="utf-8")
    assert "WANDB_IDENTITY_ENV" in text
    assert '"WANDB_RUN_ID"' in text
    assert '"WANDB_ID"' in text
    assert '"WANDB_NAME"' in text
    assert '"WANDB_RESUME"' in text
    assert "WANDB_RUN_ID={run_id}" in text
    assert "WANDB_ID={run_id}" in text
    assert "WANDB_RESUME=must" in text
    assert "WANDB_RUN_GROUP={PROBE_NAME}" in text
    assert "_with_branch_wandb_environment" in text
    assert "_assert_branch_wandb_identity" in text


def test_probe_a_impl_disables_hf_publication() -> None:
    text = IMPL.read_text(encoding="utf-8")
    assert "_publish_latest_to_bucket" not in text
    assert "_prepare(runtime_base)" not in text
    assert '"--remote-publish-every-steps", "0"' in text
    assert '"--best-model-repo"' in text
    assert "HF publication flags are forbidden for Probe A" in text
