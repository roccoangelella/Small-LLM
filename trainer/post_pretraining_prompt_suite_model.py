"""Run the standard post-pretraining suite from a stable HF model artifact.

Stable artifacts live at ``models/<run_id>/<checkpoint_id>`` and are selected by
``models/<run_id>/artifact.json``.  If that pointer is absent, a manually moved
checkpoint already under the canonical models namespace can be discovered by
its ``step-XXXXXXXX`` directory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from trainer import post_pretraining_prompt_suite as suite
from trainer.model_artifact import download_verified_model_artifact


def download_verified_stable_model(
    *,
    repo_id: str,
    run_id: str | None,
    token: str | None,
    revision: str | None,
    pointer_name: str,
    destination: Path,
):
    if pointer_name != "latest":
        raise RuntimeError(
            "stable models/<run_id>/ artifacts do not carry validation-best history; "
            "run this entrypoint with --pointer latest"
        )
    if run_id is None or not run_id.strip():
        raise RuntimeError(
            "stable model evaluation requires --run-id or SMALL_LLM_RUN_ID"
        )
    return download_verified_model_artifact(
        repo_id=repo_id,
        run_id=run_id.strip(),
        token=token,
        revision=revision,
        destination=destination,
    )


def main(argv: Sequence[str] | None = None) -> int:
    original = suite.download_verified_checkpoint
    suite.download_verified_checkpoint = download_verified_stable_model
    try:
        return suite.main(argv)
    finally:
        suite.download_verified_checkpoint = original


if __name__ == "__main__":
    raise SystemExit(main())
