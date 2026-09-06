# ADR 0152: Evaluate a sparse MoE variant at the 10B pretraining budget

Date: 2026-09-06
Status: Accepted

## Context

The canonical dense approximately-100M model has completed a 10B-token pretraining run. The next architecture experiment is intended to test whether replacing the dense feed-forward path with sparse Mixture-of-Experts (MoE) capacity can improve model quality without first moving to the planned larger-data 100B-token regime.

A fair MoE comparison requires distinguishing total parameters from parameters activated per token. The exact MoE geometry, routing rule, expert count, top-k, capacity factor, auxiliary/router losses, and whether the comparison is total-parameter-matched, active-parameter/FLOP-matched, or includes both controls are not decided by this ADR.

## Decision

- Set up an MoE architecture experiment for Small-LLM.
- Use the 10B-token pretraining budget for the first MoE comparison rather than the planned 100B-token run.
- Compare the MoE result against the existing canonical dense 100M/10B pretraining result using the same qualification/evaluation protocol wherever applicable.
- Do not treat the exact MoE geometry or implementation as decided yet; those choices require a separate design decision before wiring.

## Rationale

Holding the data budget at 10B reuses an already-qualified dense baseline and isolates the architecture change better than simultaneously changing architecture and training-token budget. Sparse MoE can add parameter capacity while keeping activated compute much closer to a dense baseline, but practical training cost also depends on routing overhead, expert utilization, memory/optimizer state, and hardware/kernel efficiency.

## Consequences

- The first MoE run is an architecture ablation at 10B tokens, not the project's next scaling-law endpoint.
- The comparison protocol must report both total parameter count and active parameters/FLOPs per token; a raw parameter-count comparison alone is insufficient.
- Router/expert utilization diagnostics must be included so a nominal MoE gain or regression can be interpreted correctly.
- No MoE code path is authorized by this ADR alone; the concrete routing and expert design remains pending.
