---
status: accepted
date: 2026-08-21
supersedes: null
---

# 0108 — Promote expanded R-SFT corpus to the Kaggle default

## Context and problem statement

The expansion lane completed all 8,473 curation-v2 keepers and froze a 16,716-row reasoning corpus at SHA-256 `d13052b6fc33108ec65511b790a75f6473144855059b16b55167b046f787c405`. The Kaggle `train` launcher still defaulted to the intermediate 12,306-row checkpoint corpus used by completed run `100m-2b-rsft-r0-12306-001`.

## Decision outcome

Make `artifacts/rsft-superior-instruction-r0-expanded/reasoning.jsonl` the standard production R-SFT training corpus. Pin the detached Kaggle worktree to commit `2ae60bfa135017353f39da2ef34a6124cda465dc`, which contains the completed corpus and compatible atomic builder/trainer, and SHA-validate the expanded manifest before training.

Use fresh default training run ID `100m-2b-rsft-r0-16716-001`. Keep `100m-2b-rsft-r0-12306-001` as the accepted trained-model identity for chat/evaluation until a new expanded-corpus model is actually trained and qualified.

Remove the tracked intermediate `rsft-superior-instruction-r0-checkpoint-12306` corpus from the current tree. Preserve its hash and historical location in Git history. Keep the 8,313-row baseline source corpus because it remains construction provenance for the expanded finalizer.

## Consequences

### Positive

- Minimal Kaggle `train` uses the completed dataset by default.
- Fresh training cannot silently resume the old 12,306-row trajectory.
- The repository drops the redundant intermediate production-sized JSONL while retaining reproducibility through Git history.

### Negative or limiting

- Exact reproduction of the old 12,306-row corpus now requires Git history.
- Until replacement training completes, default `train` and accepted `eval` intentionally use different run IDs.

## Validation

The launcher dry run must report the expanded corpus path, fresh `100m-2b-rsft-r0-16716-001` run ID, 90/10 mixture, 32,768-target optimizer blocks, and pinned implementation/corpus commit. The manifest validator must reject row-count, schema, token-range, or SHA drift.

## Links

- [`../evidence/rsft_expanded_corpus_completion_2026-08-21.md`](../evidence/rsft_expanded_corpus_completion_2026-08-21.md)
- [`../runbooks/rsft_r0_atomic_production.md`](../runbooks/rsft_r0_atomic_production.md)
