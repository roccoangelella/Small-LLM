---
status: accepted
date: 2026-09-10
supersedes: null
owners: [Small-LLM]
---

# ADR 0172: keep experiment-specific project memory on dedicated experiment branches

## Context and problem statement

When an experimental implementation has its own long-lived Git branch, writing its branch-specific design and execution decisions into `llm_docs/` on `main` mixes experimental state with the general production trajectory and makes the authoritative memory location ambiguous.

## Decision outcome

Experiment-specific project memory must live on the dedicated experiment branch when one exists. `main` should retain only decisions and references that are genuinely repository-wide or shared across trajectories.

When a dedicated experiment branch is created, architecture decisions, run plans, qualification rules, branch-specific evidence, and implementation notes that apply only to that experiment must be committed to that branch. A later merge, promotion, or retirement must explicitly decide what, if anything, is promoted back into repository-wide memory.

## Consequences

- Branch-local experiments have a single authoritative memory location.
- `main` is less likely to accumulate stale or contradictory experimental decisions.
- Work on an experiment must begin by reading the project memory from the relevant branch rather than assuming `main` is sufficient.

## Validation

For each active dedicated experiment branch, verify that experiment-only `llm_docs/` material is absent from the current `main` tree and present on the owning branch.