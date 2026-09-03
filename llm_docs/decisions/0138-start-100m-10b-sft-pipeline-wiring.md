---
status: accepted
date: 2026-09-03
supersedes: null
---

# 0138 — Start 100M/10B SFT pipeline wiring

## Decision

Start wiring the supervised-fine-tuning pipeline for the completed 100M/10B pretraining parent.

The canonical parent identity for this wiring is:

```text
run:             100m-10b-deep-decay-from-step15500
checkpoint:      step-00076294
consumed targets: 10,000,007,168
model:           100M / d_model=512 / d_ff=1408 / 20 layers / context 2048
```

This decision authorizes infrastructure work only: profile/CLI/runtime routing, parent-resolution support, identity checks, tests, and runbook updates needed to make the 100M/10B parent a first-class SFT input.

## Scientific recipe remains undecided

Do **not** infer a 100M/10B SFT recipe from the older 100M/2B experiments. In particular, this decision does not select:

- SFT target fraction or absolute target budget;
- learning-rate peak, warmup, decay, or training horizon;
- instruction/replay mixture or source allocation;
- whether the 100M/2B capacity-aware 10% bundle policy transfers to the 10B parent;
- checkpoint-selection/promotion thresholds.

The 100M/10B launcher must fail closed rather than silently inherit a 4% or 10% scientific recipe until a later accepted decision pins that recipe.

## Parent transport constraint

The completed 100M/10B endpoint is durably available through the final deep-decay checkpoint transport. The current SFT parent loader was designed around the older live model-repository pointer or stable `models/<run_id>/artifact.json` transport used by completed 2B parents. Wiring must therefore add or establish a verified final-parent transport for step 76,294 before training can launch.

Do not substitute the strict validation-loss `best` repository for the final 10B endpoint: `best` is a separate selection policy and is not the accepted final parent identity above.

## Evaluation dependency

ADR 0137 remains in force: enlarge the SFT instruction-behavior evaluation before using subsequent SFT experiments to choose a new recipe. Infrastructure wiring for 100M/10B can proceed in parallel, but recipe selection waits for that diagnostic work.
