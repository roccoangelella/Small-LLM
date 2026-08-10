---
status: superseded
date: 2026-08-07
supersedes: null
superseded_by: 0022
---

# 0008 — Run a 500M-token final data-scaling probe for the 20M model

## Context and problem statement

The approximately-20M-parameter GDN-2 hybrid is completing its fixed approximately-100M-token pretraining run. The run has demonstrated substantial held-out language-model learning, but free generation remains weak. Before moving the main project to a larger parameter scale, the 20M model should receive one final deliberately overtrained probe so its practical capacity and learning curve are characterized rather than inferred from the 100M point alone.

The 20M model has 20,637,592 learned parameters. A 500M accepted-source-token run corresponds to approximately 24.2 source tokens per parameter, which is enough to test the model beyond a 20:1 token/parameter exposure while remaining much cheaper than a 1B-token run.

## Considered options

- Stop the 20M data-scaling study after the 100M-token run and move directly to a larger model.
- Run a fresh approximately-500M-token pretraining probe for the same 20M model.
- Run a fresh approximately-1B-token probe for the same 20M model.
- Continue the completed 100M checkpoint with additional tokens under a modified schedule.

## Decision outcome

Chosen option: **run one fresh approximately-500M-token pretraining probe for the same 20M model after the current 100M run is complete and evaluated**.

The 500M probe is the intended final data-scaling characterization of the 20M smoke model. It starts from the same seed-17 initialization policy rather than continuing the 100M checkpoint, and it receives its own finite one-pass WSD schedule derived from the completed 500M dataset manifest.

The dataset profile keeps the existing frozen source revision, tokenizer, cluster policy, context length, optimizer-block geometry, shard geometry, and remote-durability mechanism. Its finite production identity is separate from the 100M dataset and targets 500,000,000 accepted source tokens with a 450,000,000 minimum, 550,000,000 hard maximum, and 20,000,000-source-token durable producer checkpoint cadence.

## Consequences

### Positive

- The 20M model is tested beyond 20 accepted source tokens per learned parameter.
- The 100M and 500M points form a much stronger same-model data-scaling curve than the 100M point alone.
- A fresh 500M WSD schedule avoids confounding the result with continuation after the 100M run's terminal learning-rate decay.
- The experiment can reuse the proven deterministic HTTP byte-range dataset producer, fixed-manifest verification, private Kaggle publication, exact-resume training, and 250-update training durability workflow.

### Negative or limiting

- The 500M run consumes substantially more T4 time than the 100M run and will likely require exact resume across platform interruptions.
- This experiment delays the first substantive approximately-100M-parameter model.
- The result characterizes the 20M model at this data mixture and training recipe; it does not by itself isolate architecture quality from parameter capacity.

## Validation

The decision is fulfilled when:

1. the current 100M run completes and receives the frozen post-pretraining evaluation;
2. a separately identified approximately-500M-token dataset is built, fully verified, remotely durable, privately published, and round-trip verified;
3. a fresh seed-17 20M model completes the exact finite one-pass 500M WSD plan with verified checkpoint/resume identity;
4. the same frozen evaluation and generation diagnostics are run so the 100M and 500M checkpoints can be compared directly.

## Links

- [`../current/status.md`](../current/status.md)
- [`../runbooks/20m_100m_runbook.md`](../runbooks/20m_100m_runbook.md)
- [`../reference/model_geometry.md`](../reference/model_geometry.md)
- [`../research/pretraining_evaluation_targets.md`](../research/pretraining_evaluation_targets.md)
