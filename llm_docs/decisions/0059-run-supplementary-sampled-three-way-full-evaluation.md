---
status: accepted
date: 2026-08-13
supersedes: null
---

# 0059 — Run supplementary sampled three-way full evaluation

## Context and problem statement

The project already has a frozen canonical qualitative protocol in ADR 0025: greedy decoding with a global 32-new-token cap. The completed 20M/500M, 20M/2B, and 100M/2B endpoints also have directly comparable `eval_core_v1` intrinsic results. The user now wants an additional full-suite comparison that exercises stochastic generation rather than greedy decoding while keeping all three checkpoints under the same sampling parameters.

This supplementary run must not silently replace ADR 0025 or be confused with the canonical greedy behavioral gate.

## Considered options

- Keep only the canonical greedy comparison.
- Run the three full suites with differing/default sampling parameters.
- Run all three full suites with one explicitly frozen supplementary sampling configuration.

## Decision outcome

Chosen option: **run the full evaluation suite for all three completed endpoints with one shared sampled decoding configuration**:

```text
temperature: 1.0
top_k: 20
top_p: 0.9
seed: 17
samples_per_prompt: 1
questions_only: false
```

Use each prompt's native full-suite generation budget; do not apply ADR 0025's 32-token cap to this supplementary comparison.

The three endpoint identities are:

```text
20M / 500M: run_id 20m-500m-dataset-001, stable models/... artifact, pointer latest
20M / 2B:   run_id 20m-2b-dataset-001, live run/... checkpoint, pointer latest
100M / 2B:  run_id 100m-2b-data-001, stable models/... artifact, pointer latest
```

The intrinsic `eval_core_v1` metrics are independent of these sampling parameters; the changed settings affect only the qualitative prompt generation portion of the full bundle.

This decision is supplementary and **does not supersede ADR 0025**. Claims about the canonical behavioral qualification must still use the frozen greedy 32-token protocol.

## Consequences

### Positive

- The three models can be compared under a more permissive, stochastic decoding regime with identical settings.
- The run may expose knowledge that is present in the distribution but not selected by greedy argmax decoding.
- The full JSON bundles retain the intrinsic scorecard and the sampled generations together under one explicit configuration.

### Negative or limiting

- A single seeded sample per prompt remains a small behavioral probe and should not be treated as a robust benchmark.
- Sampled factual answers can differ from greedy answers without any change in model weights.
- Re-running the intrinsic portion is computationally redundant because sampling does not affect teacher-forced intrinsic metrics.
- Results from this protocol are not directly interchangeable with ADR 0025's canonical greedy behavioral evidence.

## Validation

The three output JSON bundles must record identical qualitative sampling fields (`temperature=1.0`, `top_k=20`, `top_p=0.9`, `seed=17`, one sample) and the expected checkpoint identities. Their intrinsic `eval_core_v1` metrics should match the existing same-checkpoint full-evaluation results to normal deterministic/numerical tolerance.

## Links

- [`0025-freeze-canonical-full-post-pretraining-prompt-suite.md`](0025-freeze-canonical-full-post-pretraining-prompt-suite.md)
- [`../evidence/scaling/20m_500m_20m_2b_100m_2b_full_eval_2026-08-13.md`](../evidence/scaling/20m_500m_20m_2b_100m_2b_full_eval_2026-08-13.md)
- [`../runbooks/eval_core_v1_runbook.md`](../runbooks/eval_core_v1_runbook.md)
