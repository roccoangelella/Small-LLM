from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / "kaggle" / "launch.py"


def test_deep_decay_uses_dedicated_100m_repo_binding() -> None:
    source = LAUNCH.read_text(encoding="utf-8")
    assert 'DEEP_DECAY_HF_REPO_ID = "roccoangelella/small-llm-100m-qualification"' in source
    assert 'env.get("SMALL_LLM_100M_HF_REPO_ID", "").strip()' in source
    assert 'env["SMALL_LLM_HF_REPO_ID"] = (' in source
    assert 'env["SMALL_LLM_SOURCE_COMMIT"] = _local_source_commit()' in source
    assert 'subprocess.call(_deep_decay_command(args), cwd=REPO, env=env)' in source
    assert '[launch] deep-decay hf_repo=' in source
    assert 'source_commit=' in source


def test_deep_decay_dry_run_does_not_require_a_clean_checkout() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(LAUNCH),
            "deep-decay",
            "--model",
            "100M",
            "--tokens",
            "10B",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"execution": "kaggle_dual_t4_ddp_block64"' in result.stdout
    assert "controlling Small-LLM checkout is dirty" not in result.stderr
