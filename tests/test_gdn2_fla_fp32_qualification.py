"""Network-free regression tests for the FLA GDN-2 FP32 qualification gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "kaggle" / "run_gdn2_fla_fp32_qualification.py"
SPEC = importlib.util.spec_from_file_location("gdn2_fla_fp32_qualification", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
qualification = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualification)


def row(
    decay: float,
    *,
    passed: bool,
    reference_valid: bool = True,
    candidate_failure: bool = False,
) -> dict[str, object]:
    return {
        "log_decay": decay,
        "passed": passed,
        "reference_valid": reference_valid,
        "candidate_failure": candidate_failure,
        "output_ok": passed,
        "candidate_bad_gradients": ["x(nonfinite=1)"] if candidate_failure else [],
    }


def test_summary_does_not_call_reference_failure_a_fla_failure() -> None:
    summary = qualification.summarize_mode(
        [
            row(-0.25, passed=True),
            row(-0.5, passed=False, reference_valid=False),
        ]
    )

    assert summary["passing"] == [-0.25]
    assert summary["failing"] == [-0.5]
    assert summary["invalid_reference"] == [-0.5]
    assert summary["candidate_failing"] == []
    assert summary["reproduced_candidate_failure"] is False
    assert summary["all_references_valid"] is False


def test_summary_records_candidate_specific_failure_only_with_valid_reference() -> None:
    summary = qualification.summarize_mode(
        [
            row(-0.25, passed=True),
            row(-0.75, passed=False, candidate_failure=True),
        ]
    )

    assert summary["invalid_reference"] == []
    assert summary["candidate_failing"] == [-0.75]
    assert summary["reproduced_candidate_failure"] is True
    assert summary["first_candidate_failure"]["log_decay"] == -0.75


def test_verdict_is_invalid_when_reference_is_nonfinite() -> None:
    baseline = qualification.summarize_mode(
        [row(-0.5, passed=False, reference_valid=False)]
    )
    fp32 = qualification.summarize_mode([row(-0.5, passed=True)])

    assert qualification.verdict(baseline, fp32).startswith("INVALID:")


def test_verdict_requires_candidate_specific_baseline_failure() -> None:
    baseline = qualification.summarize_mode([row(-0.25, passed=True)])
    fp32 = qualification.summarize_mode([row(-0.25, passed=True)])

    assert qualification.verdict(baseline, fp32).startswith("INCONCLUSIVE-BUT-PASSING:")


def test_verdict_passes_only_when_baseline_candidate_fails_and_fp32_passes() -> None:
    baseline = qualification.summarize_mode(
        [
            row(-0.25, passed=True),
            row(-0.75, passed=False, candidate_failure=True),
        ]
    )
    fp32 = qualification.summarize_mode(
        [row(-0.25, passed=True), row(-0.75, passed=True)]
    )

    assert qualification.verdict(baseline, fp32).startswith("PASS:")


def test_verdict_fails_when_fp32_candidate_still_fails() -> None:
    baseline = qualification.summarize_mode(
        [row(-0.75, passed=False, candidate_failure=True)]
    )
    fp32 = qualification.summarize_mode(
        [row(-0.75, passed=False, candidate_failure=True)]
    )

    assert qualification.verdict(baseline, fp32).startswith("FAIL:")


def test_production_runtime_pin_matches_qualified_fla_release() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    optional = project["project"]["optional-dependencies"]
    expected = f"fla-core=={qualification.FLA_VERSION}"

    assert expected in optional["model"]
    assert expected in optional["fla"]
    adapter_source = (ROOT / "model" / "gdn2_fla.py").read_text(encoding="utf-8")
    assert f'FLA_CORE_VERSION = "{qualification.FLA_VERSION}"' in adapter_source
