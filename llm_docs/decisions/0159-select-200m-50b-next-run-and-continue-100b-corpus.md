---
status: superseded
date: 2026-09-08
supersedes: 0153
superseded_by: 0160
owners: [Small-LLM]
---

# ADR 0159: select 200M/50B as the next pretraining run and continue the 100B corpus build

## Context and problem statement

ADR 0153 selected approximately 100M parameters trained to 100B target tokens as the next major scaling experiment. Before that trajectory was launched, the scaling plan changed: the next run should increase model capacity to approximately 200M parameters while using a 50B-token training horizon.

The ongoing 100B dataset construction remains valuable independently of the immediate training horizon. Stopping it at 50B would discard useful corpus-building work and remove a ready larger-data asset for later scaling or continuation experiments.

## Considered options

- Keep ADR 0153 and run approximately 100M/100B.
- Stop corpus construction at 50B and run approximately 200M/50B.
- Run approximately 200M/50B while still completing the already-authorized 100B corpus artifact.

## Decision outcome

Chosen option: **the next major pretraining trajectory will be approximately 200M parameters trained on 50B target tokens, while construction of the full 100B dataset continues to completion.**

This supersedes ADR 0153 only with respect to the next model/token scaling target. ADR 0158's decision to prebuild and publish the complete `100b-b64-dataset-001` corpus remains in force.

The exact 200M architecture, parameter count, learning-rate schedule, optimizer/scheduler details, update geometry, checkpoint cadence, and exact mapping of the 50B training horizon onto the completed 100B corpus are not frozen by this ADR. They must be planned and reviewed before implementation.

No production 200M launcher is authorized by this decision alone.

## Consequences

### Positive

- The next experiment scales both model capacity and data relative to the completed 100M/10B endpoint.
- The full 100B corpus remains available for later continuation, larger-model, or controlled data-scaling experiments.
- Dataset construction is decoupled from the immediate 50B training horizon rather than being restarted for each scale decision.

### Negative or limiting

- The 200M/50B result will not isolate pure data scaling or pure parameter scaling relative to 100M/10B.
- A new model geometry requires fresh memory/throughput qualification on the intended providers.
- The 50B run contract cannot be wired by simply reusing the now-superseded 100M/100B scale assumptions.

## Validation

Before a long run is launched:

1. freeze and count the exact approximately-200M architecture;
2. freeze the optimizer and LR schedule over the exact 50B target-token horizon;
3. freeze block/microbatch/update geometry and provider execution constraints;
4. define the deterministic 50B consumption contract against `100b-b64-dataset-001`;
5. pass provider CPU-stage and live GPU smoke tests with exact-resume verification.

## Links

- `llm_docs/decisions/0153-select-100m-100b-as-next-pretraining-scale-target.md`
- `llm_docs/decisions/0158-prebuild-100b-in-public-hf-bucket.md`
- `llm_docs/reference/model_geometry.md`
- `llm_docs/reference/model_architecture.md`
- `llm_docs/reference/optimizer_strategy.md`
