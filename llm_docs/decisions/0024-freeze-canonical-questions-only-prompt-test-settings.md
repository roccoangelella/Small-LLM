---
status: superseded
date: 2026-08-10
superseded_by: 0025
---

# 0024 — Freeze canonical questions-only post-pretraining prompt-test settings

## Context and problem statement

The 20M / 500M post-pretraining questions-only qualitative run was launched with an explicit deterministic decoding configuration. Future model-scale and data-scale comparisons need to reuse the same generation settings so observed output differences are attributable to the checkpoint/model rather than to changes in decoding.

This decision applied specifically to the repository's `trainer.post_pretraining_prompt_suite` **questions-only qualitative generation mode**. It does not replace `eval_core_v1`, teacher-forced validation, benchmark-specific settings, or intentionally separate decoding ablations.

## Decision outcome

For canonical questions-only post-pretraining comparison, the settings recorded here were:

```text
pointer: best
device: cuda
precision: fp16
temperature: 0.0
top_p: 1.0
top_k: 0
seed: 17
samples_per_prompt: 1
questions_only: true
max_new_tokens: 32
trace_top_tokens: 0
```

Operational identity fields were not frozen across experiments and had to be supplied explicitly for the checkpoint being tested:

```text
repo_id: experiment/checkpoint repository identity
run_id: exact training run identity
output_json: experiment-specific artifact path
```

The Hugging Face secret remained read through `HF_TOKEN`; all non-secret test parameters were to be passed explicitly on the CLI rather than relying on environment defaults.

For CUDA GDN-2 checkpoints using the qualified production backend, the evaluation environment must provide the same qualified FLA core dependency:

```text
fla-core==0.5.2
```

## Supersession

This ADR is superseded by ADR 0025 because the user clarified that the reusable canonical post-pretraining comparison should run the **full qualitative prompt suite**, not only the questions subset. The deterministic decoding values remain useful historical context, but `questions_only: true` is no longer the canonical comparison protocol.

## Links

- [`0025-freeze-canonical-full-post-pretraining-prompt-suite.md`](0025-freeze-canonical-full-post-pretraining-prompt-suite.md)
- [`../runbooks/post_pretraining_prompt_suite.md`](../runbooks/post_pretraining_prompt_suite.md)
- [`0002-freeze-eval-core-v1-and-unified-cli.md`](0002-freeze-eval-core-v1-and-unified-cli.md)
