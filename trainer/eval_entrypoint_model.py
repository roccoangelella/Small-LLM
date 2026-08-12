"""Run the self-provisioning full evaluator from a stable HF model artifact.

This preserves the frozen eval_core_v1 evaluator unchanged and replaces only its
checkpoint transport. Stable completed models are resolved from
``models/<run_id>/artifact.json`` (or the newest compatible step directory) and
verified with their native ``local_manifest.json`` before model state is read.
"""
from __future__ import annotations

from typing import Sequence

from trainer import eval_entrypoint, eval_suite
from trainer.post_pretraining_prompt_suite_model import download_verified_stable_model


def main(argv: Sequence[str] | None = None) -> int:
    original = eval_suite.download_verified_checkpoint
    eval_suite.download_verified_checkpoint = download_verified_stable_model
    try:
        return eval_entrypoint.main(argv)
    finally:
        eval_suite.download_verified_checkpoint = original


if __name__ == "__main__":
    raise SystemExit(main())
