"""Run eval_core against a live HF Storage Bucket checkpoint."""

from __future__ import annotations

from trainer import eval_entrypoint, eval_suite, post_pretraining_prompt_suite
from trainer.post_pretraining_prompt_suite_bucket import download_verified_bucket_checkpoint


def main() -> int:
    post_pretraining_prompt_suite.download_verified_checkpoint = download_verified_bucket_checkpoint
    eval_suite.download_verified_checkpoint = download_verified_bucket_checkpoint
    return eval_entrypoint.main()


if __name__ == "__main__":
    raise SystemExit(main())
