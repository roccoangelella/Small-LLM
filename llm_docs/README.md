# Small-LLM project memory

This directory is the repository's system of record and uses progressive disclosure: high-freshness files stay small while detailed contracts, history, research, evidence, and execution procedures live one link deeper.

## Read in this order

1. [`current/status.md`](current/status.md) — verified present state.
2. [`current/roadmap.md`](current/roadmap.md) — immediate gates, priorities, and open decisions.
3. [`decisions/README.md`](decisions/README.md) — accepted and superseded ADRs.
4. Open only the relevant reference, runbook, plan, research note, or evidence record.

## Structure

| directory | purpose | lifecycle |
|---|---|---|
| [`current/`](current/status.md) | what is true now / what happens next | only `status.md` and `roadmap.md`; aggressively refreshed |
| [`decisions/`](decisions/README.md) | durable choices and why | supersede by new ADR; do not rewrite history |
| [`reference/`](reference/README.md) | current technical contracts | update when implementation/accepted contract changes |
| [`runbooks/`](runbooks/README.md) | executable/reproducible procedures | active or completed-stage reproduction |
| [`research/`](research/README.md) | investigation and unresolved synthesis | not authorization by itself |
| [`plans/`](plans/README.md) | complex resumable execution work | completed plans become historical traces |
| [`evidence/`](evidence/README.md) | measured observations/incidents/results | preserve; add new evidence instead of cleaning history |
| [`archive/`](archive/README.md) | superseded historical material | never current authorization |

## Current reference documents

- [`reference/project_overview.md`](reference/project_overview.md)
- [`reference/model_architecture.md`](reference/model_architecture.md)
- [`reference/model_geometry.md`](reference/model_geometry.md)
- [`reference/gdn2_chunkwise_training.md`](reference/gdn2_chunkwise_training.md)
- [`reference/gdn2_fla_backend.md`](reference/gdn2_fla_backend.md)
- [`reference/dataset_and_tokenization.md`](reference/dataset_and_tokenization.md)
- [`reference/100m_10b_incremental_dataset.md`](reference/100m_10b_incremental_dataset.md)
- [`reference/training_system.md`](reference/training_system.md)
- [`reference/optimizer_strategy.md`](reference/optimizer_strategy.md)
- [`reference/training_and_evaluation.md`](reference/training_and_evaluation.md)
- [`reference/eval_core_v1_design.md`](reference/eval_core_v1_design.md)
- [`reference/post_training_sft.md`](reference/post_training_sft.md)

## Current operational entry points

- [`runbooks/100m_10b_beam.md`](runbooks/100m_10b_beam.md)
- [`runbooks/100m_10b_incremental_modal.md`](runbooks/100m_10b_incremental_modal.md)
- [`runbooks/modal_training_launcher.md`](runbooks/modal_training_launcher.md)
- [`runbooks/unified_kaggle_launcher.md`](runbooks/unified_kaggle_launcher.md)
- [`runbooks/eval_core_v1_runbook.md`](runbooks/eval_core_v1_runbook.md)
- [`runbooks/post_pretraining_prompt_suite.md`](runbooks/post_pretraining_prompt_suite.md)
- [`runbooks/sft_s0_runbook.md`](runbooks/sft_s0_runbook.md)
- [`runbooks/rsft_r0_atomic_production.md`](runbooks/rsft_r0_atomic_production.md)

Completed scaling-stage reproduction procedures remain indexed from [`runbooks/README.md`](runbooks/README.md).

## Memory-writing contract

- Verified present facts go in `current/status.md`.
- Immediate priorities/gates/open decisions go in `current/roadmap.md`.
- A user-approved durable choice creates or supersedes an ADR.
- Current implementation details/invariants go in `reference/`.
- Commands/recovery procedures go in `runbooks/`.
- Measurements and observed incidents go in `evidence/`.
- Superseded operational material goes in `archive/`.
- Prefer links over duplicating the same detailed facts across layers.
- Garden memory when an experiment or architecture/operational plan changes lifecycle.

Default precedence when documents conflict:

```text
current status -> accepted ADR -> current reference/runbook -> evidence -> research -> archive/journals
```

The governance rationale is recorded in ADR 0031 and the project-memory research under `research/`.
