---
status: current
last_reviewed: 2026-08-06
---

# Current project status

## Active experiment

The approximately-20M-parameter GDN-2 hybrid remains authorized for one pass over the fixed approximately-100M-token dataset.

The corrected run passed the former validation-memory boundary and completed optimizer update 1,138. The next forward pass failed because the correctness-first fixed-size chunkwise GDN-2 backend produced a non-finite intermediate under strong but valid decay. The preceding loss, gradient, throughput, memory, and FP16-overflow telemetry did not show model-wide divergence.

The repository now contains an adaptive numerical repair. Assembled GDN layers retain configured maximum chunk size 32 but bisect only chunks whose cumulative decay span is unsafe, retrying down to one token if necessary. Model parameters, checkpoint keys, optimizer routing, and serialized configurations remain unchanged.

The next action is to rerun the normal Kaggle entry point from current `main`. The launcher will read the actual verified remote pointer, restore it when present, and resume the same W&B run. The expected latest durable boundary from the observed cadence is step 1,000, but this file does not substitute that expectation for pointer verification.

```text
model parameters: 20,637,592
architecture: gdn2_hybrid
context: 2,048
precision: FP16
optimizer: hybrid Muon + AdamW
training venue: Kaggle NVIDIA T4
experiment: one pass over the fixed approximately-100M-token dataset
training microbatch: 4 sequences
validation microbatch: 1 sequence
configured maximum GDN chunk: 32 tokens
adaptive decay-span limit: 60
validation cadence: 250 updates
local checkpoint cadence: 250 updates
verified remote publication cadence: 250 updates
repository default session cap: none within the finite plan
W&B run ID: 20m-100m-data-004
pinned recovery worktree: 38f0d5ae621d2a1bb5a0dd99c3cee17d98bbb0e1
```

This file does not guess live step or token progress. Read the active W&B run and verified remote checkpoint pointer for exact progress after launch.

## Accepted anchor

The completed 10M-token run remains the historical anchor:

```text
accepted source tokens: 10,000,662
optimizer updates: 306
final validation loss: 6.136690
final validation perplexity: 462.520157
FP16 overflow events: 0
```

## Frozen decisions affecting current work

- Continue the main GDN-2 hybrid through the 20M-model data-scaling stage.
- Do not run the matched all-attention or other mixer baseline yet.
- Revisit architecture comparisons when larger model versions are reached.
- Use adaptive numerical subchunking rather than globally shrinking the configured GDN chunk or clamping decay.
- Use the permanent stratified `eval_core_v1` fast/full suites and retain the existing prompt answers in the unified evaluator.
- Attempt the complete remaining 100M-token one-pass schedule in one Kaggle invocation by default.
- Validate, checkpoint locally, and publish a verified remote checkpoint every 250 successful updates.

## Evaluation state

The repository contains:

```text
small-llm-eval-data build|verify
small-llm-eval fast|full
```

The code, manifest contract, streaming metrics, prompt integration, and offline tests are implemented. The production `eval_core_v1` corpus still needs to be built and its fast/full runtime measured on the T4 before it becomes an accepted evaluation artifact.

The ordinary held-out training validation path uses `torch.inference_mode()` and a dedicated one-sequence microbatch to prevent full-vocabulary evaluation OOM on the T4.

## Current source of truth

- Experiment procedure: [`../runbooks/20m_100m_runbook.md`](../runbooks/20m_100m_runbook.md)
- GDN-2 incident evidence: [`../evidence/20m_100m/gdn2_nonfinite_step_1138_2026-08-06.md`](../evidence/20m_100m/gdn2_nonfinite_step_1138_2026-08-06.md)
- Adaptive GDN-2 decision: [`../decisions/0005-adapt-gdn2-chunks-to-decay-span.md`](../decisions/0005-adapt-gdn2-chunks-to-decay-span.md)
- Validation OOM evidence: [`../evidence/20m_100m/validation_oom_step_500_2026-08-06.md`](../evidence/20m_100m/validation_oom_step_500_2026-08-06.md)
- Revised durability decision: [`../decisions/0004-run-100m-in-one-session-with-250-step-durability.md`](../decisions/0004-run-100m-in-one-session-with-250-step-durability.md)
- Evaluation procedure: [`../runbooks/eval_core_v1_runbook.md`](../runbooks/eval_core_v1_runbook.md)
- Model contract: [`../reference/model_architecture.md`](../reference/model_architecture.md)
- GDN-2 execution contract: [`../reference/gdn2_chunkwise_training.md`](../reference/gdn2_chunkwise_training.md)
- Dataset contract: [`../reference/dataset_and_tokenization.md`](../reference/dataset_and_tokenization.md)
- Durable decisions: [`../decisions/README.md`](../decisions/README.md)
