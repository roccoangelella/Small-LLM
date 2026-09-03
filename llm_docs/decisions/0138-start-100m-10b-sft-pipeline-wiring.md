---
status: accepted
date: 2026-09-03
supersedes: null
---

# 0138 — Start 100M/10B SFT pipeline wiring

## Context and problem statement

The completed 100M/10B pretraining trajectory is now the parent we want to make available to the SFT system. Its accepted endpoint identity is:

```text
run:              100m-10b-deep-decay-from-step15500
checkpoint:       step-00076294
consumed targets: 10,000,007,168
model:            100M / d_model=512 / d_ff=1408 / 20 layers / context 2048
```

The existing 100M SFT launcher/runtime is specialized around the older 100M/2B parent. In particular, its parent transport assumes the 2B-style live/stable model-repository path, while the accepted final 10B endpoint is retained through the verified rolling Hugging Face Storage Bucket `latest` transport. The SFT pipeline therefore cannot safely treat `--tokens 10B` as a simple alias for the existing 2B profile.

ADR 0137 also remains in force: the instruction-behavior evaluation must be enlarged before subsequent SFT experiments are used to choose a new scientific recipe.

## Considered options

- Reuse the 100M/2B profile and silently inherit its 4% or 10% SFT recipe.
- Point SFT at the separate strict validation-loss `best` model repository instead of the accepted final 10B endpoint.
- Register the exact 100M/10B parent as a first-class profile, add verified final-parent transport support, and keep training actions fail-closed until a separate recipe decision is accepted.

## Decision outcome

Start wiring the supervised-fine-tuning pipeline for the completed 100M/10B pretraining parent.

This decision authorizes infrastructure work only: profile/CLI/runtime routing, parent-resolution support, identity checks, tests, and runbook updates needed to make the 100M/10B parent a first-class SFT input.

The canonical parent is the final `step-00076294` endpoint above. Do not substitute the strict validation-loss `best` repository: `best` is a separate selection policy and is not the accepted final parent identity.

Do **not** infer a 100M/10B SFT recipe from the older 100M/2B experiments. This decision does not select the SFT target fraction or absolute target budget, learning-rate schedule, instruction/replay allocation, transfer of the 100M/2B capacity-aware 10% policy, or checkpoint-promotion thresholds.

## Consequences

The 100M/10B launcher must fail closed rather than silently inherit a 4% or 10% scientific recipe until a later accepted decision pins that recipe.

Wiring must establish a verified `latest` Storage Bucket parent path for the final step-76,294 checkpoint before training can launch. Existing 100M/2B behavior must remain unchanged.

Infrastructure work may proceed in parallel with ADR 0137 evaluation design, but recipe selection waits for that diagnostic work.
