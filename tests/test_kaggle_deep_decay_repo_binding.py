from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / "kaggle" / "launch.py"


def test_deep_decay_uses_dedicated_100m_repo_binding() -> None:
    source = LAUNCH.read_text(encoding="utf-8")
    assert 'DEEP_DECAY_HF_REPO_ID = "roccoangelella/small-llm-100m-qualification"' in source
    assert 'env.get("SMALL_LLM_100M_HF_REPO_ID", "").strip()' in source
    assert 'env["SMALL_LLM_HF_REPO_ID"] = (' in source
    assert 'subprocess.call(_deep_decay_command(args), cwd=REPO, env=env)' in source
    assert '[launch] deep-decay hf_repo=' in source
