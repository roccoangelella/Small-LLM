---
status: accepted
date: 2026-08-10
supersedes: null
---

# 0032 — Scale SFT budget with pretraining and qualify on 500M first

## Context and problem statement

The reusable SFT implementation was originally designed around an approximately-20M model after a 100M-token pretraining run, including a historical fixed 4M loss-bearing-target S0 budget. The project has since completed a 500M-token pretraining trajectory and is training a fresh 2B-token trajectory on the same model geometry.

The user decided that post-training should no longer inherit the old 100M-specific fixed budget. SFT exposure should scale with the amount of pretraining, while the completed 500M checkpoint is immediately available to qualify the end-to-end SFT pipeline before the stronger 2B checkpoint is ready.

## Considered options

- Keep the historical fixed 4M SFT target-token budget for every base checkpoint.
- Defer all SFT work until the 2B pretraining run completes.
- Qualify the pipeline on the 500M checkpoint, then switch to the 2B checkpoint as soon as it is ready, with SFT target-token budget equal to 4% of the corresponding pretraining token budget.

## Decision outcome

Chosen option: **qualify SFT on the completed 500M checkpoint, then switch to the 2B checkpoint as soon as it is ready; scale SFT loss-bearing target tokens at 4% of pretraining tokens.**

Current nominal planning points are therefore:

```text
500M pretraining -> approximately 20M SFT loss-bearing target tokens
2B pretraining   -> approximately 80M SFT loss-bearing target tokens
```

The exact finite SFT horizon for each run must be derived from the verified completed parent pretraining token count and frozen in the resulting immutable SFT manifest rather than hard-coded from the nominal label alone.

The existing SFT unit remains **loss-bearing target tokens**. For instruction records these are supervised assistant-content and turn-termination targets; for replay records they are valid next-token targets.

The previously accepted overall instruction/replay policy remains unchanged unless a later ADR supersedes it. The current source-level instruction allocation also remains unchanged for the first qualification/production comparison rather than being redesigned simultaneously:

```text
75.0% smol-magpie-ultra-short
10.0% smol-contraints
 7.5% smollm-rewrite-30k
 7.5% smol-summarize-20k
```

These shares are within the instruction portion and are measured by loss-bearing target tokens.

The train/validation/test identity split is frozen at:

```text
train:       95.0%
validation:   2.5%
test:         2.5%
```

All derivatives of the same original conversation, prompt family, source document, or synthetic seed must remain in one split. The split identity must remain identical when comparing the 500M-parent and 2B-parent SFT runs so the parent pretraining checkpoint is the controlled change.

The 500M run is a qualification run for SFT correctness, stability, behavior acquisition, checkpoint/resume, and evaluation. It does not need to remain the selected post-trained model after the 2B parent checkpoint is available. The first qualified 2B-parent SFT run should reuse the frozen SFT dataset identities/policies wherever compatible so the comparison isolates the stronger parent pretraining state.

## Consequences

### Positive

- SFT exposure scales with the amount of upstream learning instead of remaining tied to an obsolete 100M-era constant.
- The completed 500M checkpoint can expose pipeline, data, masking, checkpoint, and evaluation defects before the 2B checkpoint is available.
- The 500M-versus-2B SFT comparison can hold architecture, split policy, source mixture, template, objective, and post-training machinery fixed while changing the parent pretraining state.
- A larger 2B-parent SFT budget avoids assuming that the first tiny engineering qualification budget is sufficient for a more heavily pretrained model.

### Negative or limiting

- Approximately 20M and 80M SFT target-token horizons are substantially longer than the historical 4M S0 qualification concept, so checkpoint/evaluation cadence and forgetting behavior must be re-qualified rather than copied blindly from that historical packet.
- Four percent is an experimental scaling policy, not a claim that 4% is globally optimal for future model sizes or data regimes.
- The larger SFT horizon increases the importance of pretraining replay and explicit base-retention measurement.

## Validation

Before the 2B-parent SFT run is treated as production-qualified:

1. Build and verify the immutable identity-safe 95/2.5/2.5 SFT dataset split.
2. Run the same SFT data/template/objective implementation on the 500M parent checkpoint.
3. Demonstrate finite FP16 training, exact target-token accounting, intentional checkpoint/resume equivalence, and correct next-block restoration.
4. Evaluate instruction behavior and the unchanged base `eval_core_v1` scorecard before and during SFT.
5. Use the 500M qualification evidence to freeze any remaining SFT-specific model-selection/retention thresholds and hardware/evaluation cadence that cannot be inherited directly from pretraining.
6. Switch to the 2B parent checkpoint as soon as that checkpoint has completed and passed its own post-pretraining qualification.

## Links

- [`../reference/post_training_sft.md`](../reference/post_training_sft.md)
- [`../current/status.md`](../current/status.md)
- [`../current/roadmap.md`](../current/roadmap.md)
- [`../archive/post_training_s0_2026-08-06/README.md`](../archive/post_training_s0_2026-08-06/README.md)
- [`0023-run-2b-20m-probe-via-vps-kaggle-dataset.md`](0023-run-2b-20m-probe-via-vps-kaggle-dataset.md)
- [`0027-use-500m-schema-gains-to-justify-fixed-20m-token-scaling-through-2b.md`](0027-use-500m-schema-gains-to-justify-fixed-20m-token-scaling-through-2b.md)
