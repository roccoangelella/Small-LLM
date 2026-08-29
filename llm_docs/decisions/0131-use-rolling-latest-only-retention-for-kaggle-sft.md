---
status: accepted
date: 2026-08-29
supersedes: detached-training implementation pins in 0108, 0123, and 0126
---

# ADR 0131: use rolling latest-only retention for Kaggle SFT

## Context and problem statement

An authenticated Hugging Face storage audit found approximately 102.2 GB of
private storage on the free 100-GB account. Git-backed model LFS objects
accounted for approximately 77.0 GB, private dataset repositories for 4.3 GB,
and Storage Buckets for 20.9 GB.

The canonical Kaggle SFT, scaled-SFT, and R-SFT command builders published an
approximately-914-MB `trainer_state.pkl` at every durability boundary but did
not request rolling retention. The SFT-specific trainer CLI also did not expose
or invoke the repository cleanup path already used by pretraining. As a result,
the 100M repository head retained dozens of independently resumable
intermediate checkpoints across SFT runs. Squashing Git history could not
remove those files because they remained referenced by the current tree.

## Considered options

- Retain every intermediate SFT checkpoint and increase the storage plan.
- Keep the current publisher and run periodic manual destructive cleanup.
- Retain only the newest verified checkpoint within each active SFT run and
  automatically super-squash history after pruning.

## Decision outcome

Chosen option: **retain only the newest verified remote checkpoint for each
canonical Kaggle SFT or R-SFT run**, because exact resume requires the durable
`latest.json` checkpoint, not every superseded optimizer boundary.

Every canonical SFT command passes `--remote-rolling-latest-only`. The
SFT-specific trainer accepts that option and, only after the new checkpoint
tree and two-phase pointer are durable, reuses the shared repository cleanup to:

1. delete older checkpoint directories under the same run ID;
2. remove that run's obsolete `best.json` pointer if present;
3. super-squash the target branch;
4. re-read and verify that `latest.json` still names the new checkpoint.

Cleanup does not delete other run namespaces or stable `models/...` artifacts
present in the repository head. The base SFT, 100M/2B scaled-SFT training, and
R-SFT detached worktrees are pinned to implementation commit
`184adccc1c12437046594ac674bc8d61eb710125`.

## Consequences

### Positive

- Each active SFT trajectory consumes approximately one remote trainer state
  instead of one per publication boundary.
- A failed upload cannot prune the previous durable checkpoint because cleanup
  begins only after verified two-phase publication.
- Resume, checkpoint identity, data order, optimizer state, and scientific
  training behavior are unchanged.

### Negative or limiting

- Older remote checkpoints within the same run are no longer available for
  rollback after cleanup; same-session local checkpoints remain unaffected.
- Super-squash is irreversible and applies to repository history, although all
  files still referenced by other run IDs or stable artifacts remain present.
- Hugging Face may take up to 36 hours to reflect reclaimed history storage in
  the account quota.

## Validation

- Focused unit coverage verifies SFT parser support, all three canonical command
  builders, and same-run-only checkpoint deletion before super-squash.
- The focused unittest suite passes 43 tests.
- The canonical R-SFT dry-run includes `--remote-rolling-latest-only`.
- Python compilation and `git diff --check` pass.

## Links

- [`../reference/training_system.md`](../reference/training_system.md)
- [`../runbooks/sft_s0_runbook.md`](../runbooks/sft_s0_runbook.md)
- [`../runbooks/rsft_r0_atomic_production.md`](../runbooks/rsft_r0_atomic_production.md)
