---
status: accepted
date: 2026-08-10
supersedes: 0024
---

# 0025 — Freeze canonical full post-pretraining prompt-suite settings

## Context and correction

ADR 0024 froze the deterministic settings for a questions-only run after the user's request was interpreted too narrowly. The user clarified that the reusable post-pretraining comparison should run the **full qualitative prompt suite**, not only the question subset.

The canonical comparison therefore includes every prompt category exposed by `trainer.post_pretraining_prompt_suite`: continuations, structured/pattern prompts, dialogue, and general-knowledge questions.

## Decision outcome

For every canonical full post-pretraining qualitative comparison, keep the following settings fixed unless a later ADR explicitly changes the protocol:

```text
pointer: best
device: cuda
precision: fp16
temperature: 0.0
top_p: 1.0
top_k: 0
seed: 17
samples_per_prompt: 1
questions_only: false
max_new_tokens: 32
trace_top_tokens: 0
```

`questions_only: false` means the `--questions-only` CLI flag must be omitted.

Operational identity fields are experiment-specific and must be supplied explicitly:

```text
repo_id: experiment/checkpoint repository identity
run_id: exact training run identity
output_json: experiment-specific artifact path
```

The Hugging Face secret remains read through `HF_TOKEN`; all non-secret test parameters should be explicit on the CLI rather than inherited from environment defaults.

For CUDA GDN-2 checkpoints using the qualified production backend, the evaluation environment must provide:

```text
fla-core==0.5.2
```

## Comparison rule

Across future checkpoints, change only the checkpoint/repository identity and output artifact path. Keep the full prompt set and deterministic decoding settings fixed so qualitative differences are attributable to the model/checkpoint rather than prompt selection or sampling noise.

A different decoding setup or a questions-only subset may still be run as a supplementary diagnostic, but it is not the canonical full-suite comparison.

## Origin

This ADR corrects ADR 0024 after the user explicitly clarified on 2026-08-10: the desired reusable test is the **full suite**.

## Links

- [`../runbooks/post_pretraining_prompt_suite.md`](../runbooks/post_pretraining_prompt_suite.md)
- [`0024-freeze-canonical-questions-only-prompt-test-settings.md`](0024-freeze-canonical-questions-only-prompt-test-settings.md)
- [`0002-freeze-eval-core-v1-and-unified-cli.md`](0002-freeze-eval-core-v1-and-unified-cli.md)
