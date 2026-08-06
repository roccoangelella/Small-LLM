---
status: current
last_reviewed: 2026-08-06
---

# Current project status

## Active experiment

The approximately-20M-parameter GDN-2 hybrid remains authorized for one pass over the fixed approximately-100M-token dataset.

The first attempt completed optimizer update 500 and then failed in held-out validation because evaluation forwarded the complete 16-sequence block at once. Training telemetry before the boundary was finite and stable; the incident is classified as an evaluation-memory bug, not model divergence.

The repository now contains the validation hotfix and revised durability cadence. The next action is a corrected Kaggle launch from current `main`.

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
validation cadence: 250 updates
local checkpoint cadence: 250 updates
verified remote publication cadence: 250 updates
repository default session cap: none within the finite plan
W&B run ID: 20m-100m-data-004
pinned hotfix worktree: pending latest documentation repin
```

The user does not require recovery of the failed run's local step-250 checkpoint. The launcher may still restore a compatible verified remote checkpoint when one exists; otherwise it starts the corrected attempt from seed 17.

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

The ordinary held-out training validation path now uses `torch.inference_mode()` and a dedicated one-sequence microbatch to prevent full-vocabulary evaluation OOM on the T4.

## Current source of truth

- Experiment procedure: [`../runbooks/20m_100m_runbook.md`](../runbooks/20m_100m_runbook.md)
- Validation OOM evidence: [`../evidence/20m_100m/validation_oom_step_500_2026-08-06.md`](../evidence/20m_100m/validation_oom_step_500_2026-08-06.md)
- Revised durability decision: [`../decisions/0004-run-100m-in-one-session-with-250-step-durability.md`](../decisions/0004-run-100m-in-one-session-with-250-step-durability.md)
- Evaluation procedure: [`../runbooks/eval_core_v1_runbook.md`](../runbooks/eval_core_v1_runbook.md)
- Model contract: [`../reference/model_architecture.md`](../reference/model_architecture.md)
- Dataset contract: [`../reference/dataset_and_tokenization.md`](../reference/dataset_and_tokenization.md)
- Durable decisions: [`../decisions/README.md`](../decisions/README.md)
