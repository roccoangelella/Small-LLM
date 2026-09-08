---
status: accepted
date: 2026-09-08
supersedes: 0159
owners: [Small-LLM]
---

# ADR 0160: select 200M/100B as the next pretraining run

## Context and problem statement

ADR 0159 selected approximately 200M parameters trained on 50B target tokens while continuing construction of the full 100B corpus. Before any 200M production run was wired or launched, the scaling decision changed again: compute cost is no longer the limiting reason to stop the next model at 50B tokens.

The project already intends to finish `100b-b64-dataset-001`, so the full corpus can be used directly by the next major model rather than reserving its second half only for a future continuation. Relative to the completed approximately-100M/10B endpoint, this intentionally scales both model capacity and training data by large factors in one trajectory.

## Considered options

- Keep ADR 0159 and train approximately 200M parameters on 50B targets.
- Train approximately 200M parameters on the full 100B corpus.
- Return to the older approximately-100M/100B plan from ADR 0153.

## Decision outcome

Chosen option: **the next major pretraining trajectory will be approximately 200M parameters trained on 100B target tokens.**

This supersedes ADR 0159's 50B training horizon and, transitively, ADR 0153's approximately-100M scale target. ADR 0158 remains in force: `100b-b64-dataset-001` is still built completely in its dedicated public Hugging Face bucket and paid GPU providers consume the finished corpus rather than constructing it.

The exact approximately-200M architecture, learned parameter count, learning-rate schedule, optimizer/scheduler details, global update geometry, provider-specific microbatch slicing, checkpoint cadence, and exact block-aligned terminal target count are not frozen by this ADR. They must be planned and reviewed before implementation.

No production 200M/100B launcher is authorized by this decision alone.

## Consequences

### Positive

- The next run uses the complete corpus asset already being built instead of leaving half of it unused in the immediate trajectory.
- The experiment strongly increases both model capacity and data exposure relative to the completed 100M/10B model.
- A single 200M/100B endpoint gives a stronger capability-oriented next model than the 200M/50B plan, all else equal.

### Negative or limiting

- Training compute is approximately doubled relative to the superseded 200M/50B plan.
- The result does not isolate pure parameter scaling or pure data scaling relative to 100M/10B.
- A 100B horizon makes LR-schedule design especially important: the project must not assume that a long high-LR or weakly decaying phase is appropriate merely because the token horizon is long.
- The new geometry and long-run execution path require fresh memory, throughput, stability, checkpoint, and exact-resume qualification on the intended providers.

## Validation

Before a long run is launched:

1. freeze and count the exact approximately-200M architecture;
2. freeze the optimizer and LR schedule over the exact 100B target-token horizon, explicitly considering the measured 100M evidence that stronger LR decay improved validation learning efficiency;
3. freeze global block/update geometry and provider-specific microbatch slicing;
4. verify deterministic full-corpus consumption against the completed `100b-b64-dataset-001` manifest/READY frontier;
5. freeze checkpoint/evaluation cadence and exact-resume semantics;
6. pass provider CPU-stage and live GPU smoke tests before full dispatch.

## Links

- `llm_docs/decisions/0159-select-200m-50b-next-run-and-continue-100b-corpus.md`
- `llm_docs/decisions/0158-prebuild-100b-in-public-hf-bucket.md`
- `llm_docs/decisions/0095-decay-1e-4-to-1e-5-then-5e-6.md`
- `llm_docs/evidence/scaling/100m_10b_step17789_decay_transition_learning_efficiency_2026-08-21.md`
- `llm_docs/reference/model_geometry.md`
- `llm_docs/reference/model_architecture.md`
- `llm_docs/reference/optimizer_strategy.md`
