---
status: accepted
date: 2026-09-10
supersedes: null
owners: [Small-LLM]
---

# ADR 0172: keep MoE project memory on the dedicated MoE branch

## Context and problem statement

MoE-specific architecture and execution decisions were historically written into `llm_docs/` on `main`, even after a dedicated MoE implementation branch existed. That makes branch-local design history ambiguous and lets sparse-experiment decisions leak into the dense production branch.

## Decision outcome

All MoE-specific project memory belongs on the dedicated `moe-8e-top1` branch, not on `main`.

This includes MoE architecture decisions, router/balancing decisions, sparse-training qualification policy, experiment schedules, MoE-specific execution-layout decisions, and future MoE evidence/status documents. Dense or repository-wide references may remain outside this branch only when they are genuinely cross-project rather than MoE experiment memory.

The historical MoE ADRs that still existed on `main` are migrated into this branch and removed from the current `main` tree. Future MoE decisions must be committed here unless an explicit later decision changes the branch ownership model.

## Consequences

- The MoE branch becomes the authoritative project-memory location for the sparse experiment.
- `main` remains focused on the dense/general project trajectory rather than carrying branch-specific MoE decisions.
- When reviewing MoE status or making new MoE decisions, read the project memory from `moe-8e-top1`, not only from `main`.
- Existing Git history is not rewritten; the migration changes the current branch trees and preserves historical commits.

## Validation

Verify that every MoE-specific ADR found retroactively in the current `main` tree exists on `moe-8e-top1` and no longer exists at the same path on `main`.