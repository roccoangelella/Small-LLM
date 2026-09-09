---
status: accepted
date: 2026-09-09
supersedes: null
---

# 0169 — Reserve 100B for MoE and isolate MoE work on its branch

## Context and problem statement
The project expects at most one affordable full 100B-token MoE pretraining run. MoE experiments must therefore protect that budget and remain isolated from the dense/main development line.

## Considered options
- Keep the 100B budget for a dense run and treat MoE as exploratory.
- Reserve the 100B budget for the eventual qualified MoE.
- Develop MoE directly on `main`.
- Keep all MoE-specific work on the dedicated `moe-8e-top1` branch until an explicit merge decision.

## Decision outcome
Chosen option: **reserve 100B tokens for the eventual MoE pretraining run and perform MoE-specific code, experiments, and project-memory updates on `moe-8e-top1`, not `main`, unless explicitly changed later**.

## Consequences

### Positive
- Protects the one full MoE training budget.
- Keeps dense/main development isolated from MoE experimentation.
- Forces short qualification probes before spending the 100B-token budget.

### Negative or limiting
- The MoE branch can diverge from `main` and may require deliberate synchronization later.
- MoE-specific decisions must not be written to `main` by default.

## Validation
- MoE-specific commits and ADRs land on `moe-8e-top1`.
- The full MoE run budget remains 100B tokens and is launched only after short qualification probes.

## Links
- Branch: `moe-8e-top1`
