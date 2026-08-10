# Small-LLM project memory

This directory is the repository's system of record. It is deliberately split by purpose so a human or coding agent can load a small map first and only open the documents relevant to the task.

## Read in this order

1. [`current/status.md`](current/status.md) — verified present state.
2. [`current/roadmap.md`](current/roadmap.md) — next gates and open questions.
3. [`decisions/README.md`](decisions/README.md) — accepted and superseded ADRs.
4. The relevant reference document or runbook.

## Structure

| Directory | Question it answers | Maintenance rule |
|---|---|---|
| [`current/`](current/status.md) | What is true now? What happens next? | Short, frequently reviewed, no historical essay. |
| [`decisions/`](decisions/README.md) | Why did we choose this? | One durable decision per ADR; never silently rewrite accepted rationale. |
| [`reference/`](reference/README.md) | How is the system defined? | Current technical contracts and detailed system descriptions. |
| [`runbooks/`](runbooks/README.md) | How do I operate it? | Executable commands, prerequisites, checks, and recovery steps. |
| [`research/`](research/README.md) | What did we investigate? | Sources, comparisons, assumptions, and conclusions separated from decisions. |
| [`evidence/`](evidence/README.md) | What actually happened? | Completed results are immutable evidence. |
| [`archive/`](archive/README.md) | What used to be planned or used? | Superseded material only; commands may no longer exist. |

## Active reference documents

- [`reference/project_overview.md`](reference/project_overview.md)
- [`reference/model_architecture.md`](reference/model_architecture.md)
- [`reference/model_geometry.md`](reference/model_geometry.md)
- [`reference/dataset_and_tokenization.md`](reference/dataset_and_tokenization.md)
- [`reference/training_system.md`](reference/training_system.md)
- [`reference/optimizer_strategy.md`](reference/optimizer_strategy.md)
- [`reference/training_and_evaluation.md`](reference/training_and_evaluation.md)
- [`reference/eval_core_v1_design.md`](reference/eval_core_v1_design.md)

## Active runbooks

- [`runbooks/20m_100m_runbook.md`](runbooks/20m_100m_runbook.md)
- [`runbooks/20m_500m_runbook.md`](runbooks/20m_500m_runbook.md)
- [`runbooks/20m_1b_runbook.md`](runbooks/20m_1b_runbook.md)
- [`runbooks/eval_core_v1_runbook.md`](runbooks/eval_core_v1_runbook.md)
- [`runbooks/post_pretraining_prompt_suite.md`](runbooks/post_pretraining_prompt_suite.md)

## Memory-writing contract

- Operational facts go in `current/status.md`.
- Priorities and gates go in `current/roadmap.md`.
- A user-approved durable choice creates or supersedes one ADR.
- Implementation details go in reference docs, not in status or ADRs.
- Commands go in runbooks, not in reference essays.
- Measurements go in evidence, not in plans.
- Superseded plans move to archive.
- Update documentation in the same commit as the code or operational change it describes.

The rationale for this layout is recorded in [`research/project_memory_research.md`](research/project_memory_research.md) and ADR [`decisions/0001-use-structured-markdown-project-memory.md`](decisions/0001-use-structured-markdown-project-memory.md).
