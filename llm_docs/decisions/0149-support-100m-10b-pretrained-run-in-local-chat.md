---
status: accepted
date: 2026-09-04
---

# 0149 — Support 100M / 10B pretrained endpoint in local chat

## Context and problem statement

The active 100M / 10B deep-decay pretraining continuation (`100m-10b-deep-decay-from-step15500`) completed its full planned 10,000,007,168-token schedule at step 76,294. Under ADR 0132, rolling and final execution checkpoints are preserved in the private Hugging Face Storage Bucket (`<owner>/<name>-checkpoints`), while the separate dedicated best-model repository holds intermediate validation-loss best snapshots.

The local `chat.py` CLI supported the completed 100M / 2B pretrained baseline (`100m-2b-data-001`) via stable model artifacts (ADR 0064) and the completed 100M / 10B SFT model (ADR 0148), but had no registered pretrained profile for the completed 100M / 10B pretraining run.

The completed 100M / 10B pretraining run is needed for qualitative comparison and interactive evaluation directly from the same local CLI.

## Considered options

- Require manual publication of the completed 10B endpoint to the legacy Git-backed model repository `models/` namespace before allowing chat.
- Keep `chat.py` restricted to 100M / 2B for pretraining.
- Register `(100_000_000, 10_000_000_000)` under the `--pre-trained` stage in `chat.py` with run ID `100m-10b-deep-decay-from-step15500`, downloading and verifying the completed checkpoint directly from the private Hugging Face Storage Bucket (with fallback to model artifacts if published).

## Decision outcome

Chosen option: **register the 100M / 10B pretraining run `100m-10b-deep-decay-from-step15500` as a supported pretrained profile in `chat.py` backed by the Hugging Face Storage Bucket transport.**

The command is:

```bash
python chat.py --model_params 100M --num_tokens 10B --pre-trained
```

The profile uses `trainer.post_pretraining_prompt_suite_bucket.download_verified_bucket_checkpoint`, reads `SMALL_LLM_HF_REPO_ID` (or `SMALL_LLM_HF_CHECKPOINT_BUCKET_ID`), verifies the published checkpoint manifest and local manifest, and enforces that the checkpoint consumed the full schedule horizon (10,000,007,168 loss-bearing targets) before loading weights.

As with 100M / 2B, interactive generation uses the standard chat template for consistency, but the model remains a pretrained base model without instruction tuning.

## Consequences

### Positive

- The completed 100M / 10B pretraining endpoint can be inspected interactively with the same local CLI.
- Reuses the canonical Storage Bucket transport established by ADR 0132 without needing redundant Git/LFS commits to the model repository.
- Schedule completion check fails closed on incomplete checkpoints.
- Existing SFT, R-SFT, and 100M / 2B profiles remain untouched.

### Negative or limiting

- The 100M / 10B model is a pretrained base model, not instruction-tuned.
- The CLI remains a qualitative convenience surface rather than a high-throughput serving runtime.

## Validation

- `tests/test_chat_cli.py` verifies resolution of `100M` / `10B` to `100m-10b-deep-decay-from-step15500` with source `storage_bucket`.
- `tests/test_chat_cli.py` verifies repository resolution for `storage_bucket` and mock downloading via the bucket checkpoint helper.
- `tests/test_chat_schedule_completion.py` verifies the WSqD schedule completion horizon computation for the 100M / 10B parameters.

## Links

- [`0064-allow-stable-pretrained-artifacts-in-local-chat.md`](0064-allow-stable-pretrained-artifacts-in-local-chat.md)
- [`0132-split-latest-checkpoints-to-hf-bucket-and-best-model-to-dedicated-repo.md`](0132-split-latest-checkpoints-to-hf-bucket-and-best-model-to-dedicated-repo.md)
- [`0148-register-100m-10b-sft-for-local-chat.md`](0148-register-100m-10b-sft-for-local-chat.md)
- [`../runbooks/local_sft_chat.md`](../runbooks/local_sft_chat.md)
- [`../current/status.md`](../current/status.md)
