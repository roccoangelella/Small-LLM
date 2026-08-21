---
status: accepted
date: 2026-08-21
supersedes: null
---

# 0111 — Allow epoch count directly on production R-SFT

## Context and problem statement

The R-SFT trainer already supports exact repeated replay of immutable train blocks with logical block IDs, epoch-count-bound pipeline identity, and exact checkpoint/resume. That capability was exposed only through the historical ablation lane, while production `train` rejected `--num-epochs > 1`. The operator wants to run controlled multi-pass training on the completed 16,716-row production corpus through the normal production command.

## Decision outcome

Allow positive `--num-epochs N` directly on `kaggle/launch_r_sft.py train`. Keep production atomic-only and reuse the existing exact-block replay mechanism; do not reshuffle or rebuild the corpus between epochs.

When `--run-id` is omitted, assign an epoch-specific production identity: one epoch remains `100m-2b-rsft-r0-16716-001`; `N > 1` uses `100m-2b-rsft-r0-16716-eN-001`. Reject multi-epoch launches that explicitly reuse the one-epoch run ID, and reject the historical accepted `100m-2b-rsft-r0-12306-001` identity for expanded-corpus production training.

For the frozen 417-block bundle, two epochs are exactly 834 optimizer steps. Checkpoint/resume uses logical block IDs across the repeated stream and the pipeline identity records `num_epochs`, so one-epoch and multi-epoch checkpoints cannot cross-resume.

## Consequences

### Positive

- A two-pass production experiment is now the simple command `python kaggle/launch_r_sft.py train --model 100M --tokens 2B --num-epochs 2`.
- Run namespaces remain isolated automatically.
- Existing exact-resume semantics are reused rather than introducing a second repeat implementation.

### Negative or limiting

- Repeated epochs intentionally replay the same frozen order; they do not create fresh shuffles.
- Training cost and target exposure scale linearly with epoch count.
- A multi-epoch model remains a separate experiment until explicitly qualified/promoted.

## Validation

The two-epoch production dry run must report `atomic-production-repeat-v1`, run ID `100m-2b-rsft-r0-16716-e2-001`, `num_epochs=2`, `bundle-exact-repeat`, and `--rsft-num-epochs 2`, while preserving the expanded corpus, 100M repository binding, and checkpoint-upload safeguards.

## Links

- [`../runbooks/rsft_r0_atomic_production.md`](../runbooks/rsft_r0_atomic_production.md)
