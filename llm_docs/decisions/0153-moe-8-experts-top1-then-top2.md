# ADR 0153 — First MoE ablation uses 8 experts, top-1 first, then top-2

Date: 2026-09-06
Status: accepted

## Decision

The first Small-LLM Mixture-of-Experts experiment will keep the established 100M/10B dense run as the baseline and introduce an MoE FFN with **8 routed experts**.

The ablation order is:

1. **8 experts, top-1 routing** as the first MoE training run.
2. **8 experts, top-2 routing** as the second MoE training run.

The intent is to isolate the effect of activating a second expert while keeping the expert pool fixed. Other training/data/evaluation settings should be held constant wherever technically possible so that the top-1 vs top-2 comparison is interpretable.

## Constraints

- The experiments use the same 10B-token pretraining budget chosen for the initial MoE study.
- The dense 100M/10B model remains the reference baseline.
- Exact expert width, router formulation, balancing/capacity policy, auxiliary losses, optimizer routing, checkpoint schema, telemetry, and runtime kernels are not decided by this ADR and must be planned before implementation.
- No implementation should be wired until the user has demonstrated understanding of the chosen design details, per the project protocol.

## Rationale

Top-1 provides the simplest sparse-routing control and the lowest active expert compute. Reusing the same 8-expert pool for top-2 creates a direct second ablation that tests whether activating an additional expert per token is worth its extra active compute or whether a compute-matched expert-width adjustment is preferable.
