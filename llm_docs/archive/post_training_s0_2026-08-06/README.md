# Historical S0 / SFT design packet — 2026-08-06

This directory preserves the August 6 design-freeze work for a first S0 supervised-fine-tuning qualification on the approximately-20M model after the 100M pretraining stage.

It contains both actual user-approved choices and proposals that were explicitly still open. The packet explains the design lineage of the reusable `post_training/sft/` implementation, but it is **not current authorization for a future SFT run** and its 100M-era numeric choices must not be silently generalized to a later base checkpoint.

Current implementation facts are summarized in [`../../reference/post_training_sft.md`](../../reference/post_training_sft.md). A future production SFT run requires a new ADR that explicitly binds its parent checkpoint, dataset, objective, budget, optimizer/schedule, evaluation, and selection gates.

## Retained packet contents

- [`sft_design_freeze.md`](sft_design_freeze.md) — broad initial design/research document; many choices explicitly proposed or deferred.
- [`sft_curriculum_sequence_decision.md`](sft_curriculum_sequence_decision.md) — frozen S0/S1/S2 curriculum direction with many exact inputs left open.
- [`s0_budget_scalability_decision.md`](s0_budget_scalability_decision.md) — 4M-target and configurability decision scoped to the 20M/100M S0 concept.
- [`s0_filtering_layers_decision.md`](s0_filtering_layers_decision.md) — approved source allowlisting and deterministic hard-filter layers; semantic classification remained unfrozen.
- [`s0_training_architecture_decisions.md`](s0_training_architecture_decisions.md) — most complete later S0 architecture decision record from the packet.

## Retired redundant drafts

Two overlapping architecture snapshots were intentionally removed from the working tree during the 2026-08-10 memory cleanup because the retained comprehensive architecture record subsumes their useful content:

```text
llm_docs/s0_training_architecture_decision.md
llm_docs/s0_training_architecture_decisions_2026-08-06.md
```

Their exact pre-cleanup contents remain available in Git history at commit `4c711ed88cc6554b9d1adab045deba082e171ec9`.

## Historical interpretation

Within this packet, later/more explicit records supersede conflicting candidate defaults in earlier design documents. This local historical precedence does not turn the packet into current project policy.
