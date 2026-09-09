---
status: accepted
date: 2026-09-09
supersedes: null
---

# 0168 — Require MoE qualification before any full 100B sparse run

## Context and problem statement

The project expects the budget for a full 100B-token pretraining trajectory to be scarce enough that a second full trajectory cannot be assumed. The current eight-full-expert, dropless Top-1 MoE is useful as a systems and routing baseline, but it has not yet established that its expert topology, routing stability, load distribution, or dispatch efficiency are the best use of a one-shot 100B budget.

Recent sub-billion MoE evidence also makes the architecture choice nontrivial: expert granularity, shared expert capacity, routing topology, balancing, and execution kernels interact strongly. Therefore a full 100B sparse run must not be used as the experiment that discovers whether the MoE architecture itself is sound.

This decision does not by itself supersede ADR 0160's already-selected dense approximately-200M/100B trajectory, nor does it select a final MoE geometry. Any later choice to replace that trajectory with a sparse one requires an explicit architecture/run decision.

## Considered options

- Launch the current eight-full-expert Top-1 MoE on 100B and learn from the result.
- Treat the current MoE as a baseline and qualify competing sparse architectures at much smaller token budgets before selecting any full 100B sparse trajectory.
- Defer MoE work and retain only the dense approximately-200M/100B trajectory.

## Decision outcome

Chosen option: **treat the current MoE as a baseline and require architecture qualification before any full 100B sparse trajectory**.

The qualification must separate at least four questions: learning quality at equal active-compute/token budget, routing stability and expert utilization, systems throughput/memory, and exact checkpoint/resume stability. The full 100B run may proceed only after one MoE design is selected from bounded lower-cost evidence rather than intuition alone.

## Consequences

### Positive

- Preserves the scarce full-run budget for a design that has already demonstrated stable routing and useful learning.
- Prevents the current 8x coarse Top-1 topology from becoming the final architecture merely because it was implemented first.
- Makes systems optimization and architecture quality separately measurable.
- Allows recent sub-billion MoE evidence on granularity, shared experts, routing, and balancing to inform the final design.

### Negative or limiting

- Requires additional small qualification runs before a sparse 100B launch.
- The qualification suite itself must be designed carefully enough that short-run routing transients are not mistaken for long-run behavior.
- No final MoE topology is authorized by this ADR.

## Validation

Before a full 100B sparse launch, the selected candidate must demonstrate, against the current MoE and an appropriate dense/active-compute reference where feasible:

1. finite and stable training with exact checkpoint/resume;
2. no pathological routing collapse or persistently dormant routed capacity under the chosen balancing policy;
3. better or defensibly equal validation/qualification learning at matched consumed-token and active-compute budgets;
4. measured provider throughput and memory that make the intended long trajectory operationally viable;
5. routing/utilization telemetry sufficient to distinguish healthy specialization from accidental concentration.

## Links

- `llm_docs/decisions/0160-select-200m-100b-as-next-pretraining-run.md`
- `llm_docs/current/roadmap.md`
- MoE benchmark/research package reviewed on 2026-09-09 (external project evidence; not stored in this ADR)
