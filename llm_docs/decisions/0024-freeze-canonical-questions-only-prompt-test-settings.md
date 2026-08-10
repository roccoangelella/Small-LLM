---
status: accepted
date: 2026-08-10
---

# 0024 — Freeze canonical questions-only post-pretraining prompt-test settings

## Context and problem statement

The 20M / 500M post-pretraining questions-only qualitative run was launched with an explicit deterministic decoding configuration. Future model-scale and data-scale comparisons need to reuse the same generation settings so observed output differences are attributable to the checkpoint/model rather than to changes in decoding.

This decision applies specifically to the repository's `trainer.post_pretraining_prompt_suite` **questions-only qualitative generation mode**. It does not replace `eval_core_v1`, teacher-forced validation, benchmark-specific settings, or intentionally separate decoding ablations.

## Decision outcome

For every canonical questions-only post-pretraining comparison, keep the following settings fixed unless a later ADR explicitly changes the protocol:

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

Operational identity fields are not frozen across experiments and must be supplied explicitly for the checkpoint being tested:

```text
repo_id: experiment/checkpoint repository identity
run_id: exact training run identity
output_json: experiment-specific artifact path
```

The Hugging Face secret remains read through `HF_TOKEN`; all non-secret test parameters should be passed explicitly on the CLI rather than relying on environment defaults.

For CUDA GDN-2 checkpoints using the qualified production backend, the evaluation environment must provide the same qualified FLA core dependency:

```text
fla-core==0.5.2
```

The canonical mode is therefore greedy and deterministic. `top_p=1` and `top_k=0` are retained explicitly even though temperature zero selects argmax, so the full decoding contract is visible in commands and JSON metadata.

## Rationale

- Greedy decoding removes sampling noise from checkpoint-to-checkpoint qualitative comparison.
- Seed 17 remains fixed for metadata consistency and for any future code path where the seed is consulted.
- One sample per prompt avoids creating unequal qualitative sample budgets across runs.
- A 32-token cap keeps answers short and makes the suite substantially faster while preserving enough room for the current general-knowledge prompts.
- Disabling token tracing keeps the canonical question run focused on outputs; tracing remains a separate diagnostic when needed.
- Using the validation-selected `best` pointer keeps the comparison tied to the repository's established checkpoint-selection rule.
- Explicit CLI arguments make saved commands auditable and prevent environment defaults from silently changing the protocol.

## Comparison rule

When comparing future checkpoints with this questions-only suite, change only the fields required to identify the target experiment/checkpoint and its output artifact. Do not change decoding settings to improve a model's apparent answers within the canonical comparison.

If a different decoding setup is scientifically useful, run it as an explicitly labeled supplementary diagnostic rather than silently replacing this canonical protocol.

## Origin

The frozen settings are the exact settings used for the 20M / 500M questions-only test targeting run `20m-500m-dataset-001` and its validation-selected `best` pointer on 2026-08-10.

## Links

- [`../runbooks/post_pretraining_prompt_suite.md`](../runbooks/post_pretraining_prompt_suite.md)
- [`../current/status.md`](../current/status.md)
- [`0002-freeze-eval-core-v1-and-unified-cli.md`](0002-freeze-eval-core-v1-and-unified-cli.md)
