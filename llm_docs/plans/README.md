# Execution plans

Use this directory for self-contained plans for complex multi-step work that benefits from resumability across agents or sessions.

A plan is not project truth by itself. It records how an authorized task is being executed. Durable user choices still belong in ADRs; observed results still belong in evidence; final technical behavior still belongs in reference/runbooks/current state.

## Lifecycle

```text
plans/active/     work still being executed
plans/completed/  finished execution traces retained for history
```

Use a plan when the task spans multiple subsystems, has several validation gates, or is likely to need handoff/resume. Do not create plans for routine one-file changes.

A useful plan should be self-contained and include:

- objective and non-goals;
- relevant current constraints/ADRs;
- ordered work items;
- progress/checklist;
- discoveries that changed execution;
- validation performed and remaining gaps;
- links to resulting ADR/reference/evidence/runbook changes.

Keep active plans updated while work proceeds. Once finished, move the plan to `completed/` and ensure current project state is represented elsewhere; completed plans are historical execution traces, not active guidance.

## Completed

- [`completed/2026-08-10-project-memory-refactor.md`](completed/2026-08-10-project-memory-refactor.md)
