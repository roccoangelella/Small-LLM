---
status: accepted
date: 2026-08-07
supersedes: null
---

# 0009 — Start the 500M run at microbatch 4 with 250-step durability

## Context and problem statement

The approximately-20M-parameter model has already exercised microbatch 4 on the same Kaggle NVIDIA T4 training path during the 100M-token experiment. The 500M run is a final data-scaling probe of the same model geometry and training implementation, not a new hardware or batch-size qualification experiment.

The inherited 100M launcher normally runs two fresh eight-update probes, microbatch 1 and microbatch 4, before a fresh training run. Repeating those probes would consume startup time without changing the selected operating point. The 500M experiment must also retain the corrected durability policy in which held-out validation, local checkpointing, and verified remote checkpoint publication happen every 250 successful optimizer updates.

## Considered options

- Repeat the microbatch 1-vs-4 qualification before the fresh 500M run.
- Skip the probe and start real training immediately at microbatch 4 while retaining the ordinary numerical, memory, validation, checkpoint, and resume safeguards.
- Change the durability cadence for the longer 500M run.

## Decision outcome

Chosen option: **skip the fresh microbatch probes for the 500M experiment, start real training immediately at microbatch 4, and keep validation/local checkpoint/verified remote publication every 250 successful optimizer updates.**

The 500M launch summary must record the skipped qualification explicitly as an experiment decision with `selected_microbatch: 4` and `probe_steps_executed: 0`; it must not fabricate a passed probe result.

This decision is scoped to the 20M-model/500M-token experiment. The ordinary 100M launcher and future model/hardware qualification policy remain unchanged.

## Consequences

### Positive

- No startup time is spent replaying the already-established microbatch 1-vs-4 comparison.
- Update 1 of the fresh 500M schedule is a real training update at microbatch 4.
- The 250-step verified remote publication cadence preserves the recovery behavior proven during the 100M run.
- The audit trail distinguishes an intentionally skipped probe from a measured qualification result.

### Negative or limiting

- The 500M launch does not independently re-measure microbatch-4 headroom before update 1.
- If the Kaggle T4/runtime environment changes materially, the existing fail-closed runtime safeguards rather than a dedicated pre-run probe become the first detector.
- This choice must not be generalized automatically to a larger model, different accelerator, different precision regime, or different batch geometry.

## Validation

The 500M launcher is correct when:

1. a fresh launch produces no `microbatch-1-probe` or `microbatch-4-probe` training stages;
2. its launch summary records `status: skipped_by_experiment_decision`, `selected_microbatch: 4`, and `probe_steps_executed: 0` for microbatch qualification;
3. the real trainer command contains `--microbatch-size 4` from the first update;
4. update 250 performs held-out validation, creates a local checkpoint, and completes verified remote publication;
5. subsequent durability boundaries retain the same 250-update cadence and exact-resume contract.

## Links

- [`0008-run-500m-final-20m-data-scaling-probe.md`](0008-run-500m-final-20m-data-scaling-probe.md)
- [`../runbooks/20m_500m_runbook.md`](../runbooks/20m_500m_runbook.md)
- [`../current/status.md`](../current/status.md)
