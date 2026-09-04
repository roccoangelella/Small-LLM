---
status: accepted
date: 2026-09-04
owners: [Small-LLM]
---

# ADR 0142: ignore local agent tooling directories

## Decision

Treat repository-root agent-tooling state as local-only and ignore it in Git.

The root `.gitignore` must ignore:

- `/.agents/`, which is the casing currently present in the repository;
- `/.Agents/`, covering the explicitly requested alternate casing;
- `/.pi/`.

`AGENTS.md` remains tracked and is not affected by these directory rules.

## Rationale

These directories contain local agent/tool configuration and state rather than
model code, training data, evaluation contracts, or project-memory artifacts.
Keeping them out of normal Git change detection avoids machine- and
agent-specific churn in the project history.

## Existing tracked content

Both `.agents/` and `.pi/` already contain tracked content on `main` at the time
of this decision. Adding ignore rules does not retroactively untrack committed
files; removing those paths from the Git index is a separate repository-history
change and is not implied by this ADR.
