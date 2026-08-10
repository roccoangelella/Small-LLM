# Small-LLM project memory

This directory is the repository's system of record. It uses **progressive disclosure**: the files read on every task stay very small, while detailed contracts, history, research, evidence, and execution traces are one link deeper.

The root of `llm_docs/` is intentionally only this map.

## Read in this order

1. [`current/status.md`](current/status.md) — verified present state.
2. [`current/roadmap.md`](current/roadmap.md) — immediate gates, priorities, and open decisions.
3. [`decisions/README.md`](decisions/README.md) — accepted and superseded ADRs.
4. Open only the relevant reference document, runbook, active plan, research note, or evidence record for the task.

## Structure

| Directory | Question it answers | Lifecycle rule |
|---|---|---|
| [`current/`](current/status.md) | What is true now? What happens next? | Only `status.md` and `roadmap.md`; aggressively refreshed. |
| [`decisions/`](decisions/README.md) | Why did we choose this? | One durable choice per ADR; supersede rather than rewrite history. |
| [`reference/`](reference/README.md) | What is the current technical contract? | Authoritative implementation/system description, not chronology. |
| [`runbooks/`](runbooks/README.md) | How do I operate or reproduce it? | Executable/reproducible procedures and recovery steps. |
| [`research/`](research/README.md) | What did we investigate or still need to decide? | Sources, comparisons, assumptions, and recommendations; never authorization by itself. |
| [`plans/`](plans/README.md) | How is a complex authorized task being executed? | Self-contained active plans; completed plans become historical traces. |
| [`evidence/`](evidence/README.md) | What actually happened? | Immutable observations and measurements, preferably grouped by run/topic. |
| [`archive/`](archive/README.md) | What used to be planned, current, or operational? | Superseded/historical material only; never current authorization. |

Informal personal study notes remain in the repository-level `journals/` directory and are not authoritative project state.

## Current reference documents

- [`reference/project_overview.md`](reference/project_overview.md)
- [`reference/model_architecture.md`](reference/model_architecture.md)
- [`reference/model_geometry.md`](reference/model_geometry.md)
- [`reference/gdn2_chunkwise_training.md`](reference/gdn2_chunkwise_training.md)
- [`reference/gdn2_fla_backend.md`](reference/gdn2_fla_backend.md)
- [`reference/dataset_and_tokenization.md`](reference/dataset_and_tokenization.md)
- [`reference/training_system.md`](reference/training_system.md)
- [`reference/optimizer_strategy.md`](reference/optimizer_strategy.md)
- [`reference/training_and_evaluation.md`](reference/training_and_evaluation.md)
- [`reference/eval_core_v1_design.md`](reference/eval_core_v1_design.md)
- [`reference/post_training_sft.md`](reference/post_training_sft.md)

## Current operational entry points

- [`runbooks/unified_kaggle_launcher.md`](runbooks/unified_kaggle_launcher.md)
- [`runbooks/20m_2b_runbook.md`](runbooks/20m_2b_runbook.md)
- [`runbooks/eval_core_v1_runbook.md`](runbooks/eval_core_v1_runbook.md)
- [`runbooks/post_pretraining_prompt_suite.md`](runbooks/post_pretraining_prompt_suite.md)

Completed scaling-stage reproduction procedures remain indexed from [`runbooks/README.md`](runbooks/README.md). Superseded plans belong in archive.

## Memory-writing contract

- Verified present facts go in `current/status.md`.
- Immediate priorities, gates, and open decisions go in `current/roadmap.md`.
- A user-approved durable choice creates or supersedes one ADR.
- Current implementation details and invariants go in `reference/`.
- Commands and recovery procedures go in `runbooks/`.
- Unresolved analysis and external-source synthesis go in `research/`.
- Complex resumable execution work may use `plans/active/`; completed plans move to `plans/completed/`.
- Measurements and observed incidents go in `evidence/` and are not rewritten to make later interpretation cleaner.
- Superseded plans, completed investigation handoffs, and obsolete design packets move to `archive/`.
- Prefer links over copying the same facts into multiple layers.
- Garden memory when an experiment, investigation, architecture migration, or major plan changes lifecycle.
- Update documentation with the code/operational change it describes whenever practical.

Default precedence when documents conflict:

```text
current status -> accepted ADR -> current reference/runbook -> evidence -> research -> archive/journals
```

## Design rationale

The memory architecture and research basis are documented in:

- [`research/agent_memory_and_documentation_2026-08-10.md`](research/agent_memory_and_documentation_2026-08-10.md)
- ADR [`decisions/0031-govern-project-memory-with-progressive-disclosure.md`](decisions/0031-govern-project-memory-with-progressive-disclosure.md)
- earlier foundational research [`research/project_memory_research.md`](research/project_memory_research.md)
- ADR [`decisions/0001-use-structured-markdown-project-memory.md`](decisions/0001-use-structured-markdown-project-memory.md)
