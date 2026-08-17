# ADR 0084: Resolve SFT parents from stable model artifacts when live pointers are absent

- Status: Accepted
- Date: 2026-08-15

## Context and problem statement

The canonical 100M / 2B post-SFT full qualification failed before evaluation because parent resolution requested `run/100m-2b-data-001/best.json` from the Hugging Face qualification repository. The completed 100M / 2B pretrained parent is published under the stable model-artifact namespace `models/100m-2b-data-001/...`, so the historical live-run pointer is not guaranteed to exist.

The SFT checkpoint itself still uses the live `run/<sft_run_id>/latest.json` publication protocol while training is active, so parent and SFT transports cannot be assumed to be identical.

## Decision

`post_training.sft.checkpoints.download_parent_checkpoint` keeps the requested historical live pointer as the first resolution path. If and only if that live pointer is absent, it falls back to the verified stable model-artifact resolver in `trainer.model_artifact`.

The fallback does not run for checkpoint-integrity or other transport failures. This prevents a corrupt or invalid live checkpoint from being silently replaced by a different artifact.

The 100M / 2B Kaggle SFT profile is pinned to implementation commit `ca16b22905ebedc5925ab0abb9c40125254f1e1c`, which includes the resolver change and regression coverage.

## Consequences

- `python kaggle/launch_sft.py eval --model 100M --tokens 2B --suite full` can resolve the completed pretrained parent from `models/100m-2b-data-001/...` when `run/100m-2b-data-001/best.json` is absent.
- Historical SFT parents that still expose `run/<run_id>/best.json` retain their previous behavior.
- The SFT candidate remains resolved through its own live `latest` pointer.
- Stable artifacts remain verified through their native `local_manifest.json` contract.
- Integrity failures fail closed instead of triggering fallback.
