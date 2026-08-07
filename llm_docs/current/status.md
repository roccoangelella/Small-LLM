---
status: current
last_reviewed: 2026-08-07
---

# Current project status

## Active experiment

The approximately-20M-parameter GDN-2 hybrid remains authorized for one pass over the fixed approximately-100M-token dataset.

The adaptive GDN-2 repair passed the former update-1,138 failure boundary. The resumed run then completed optimizer update 1,497 and failed on the next prepared block because the trainer allowed only three FP16 overflow retries. Four scaled-gradient overflows caused termination immediately after the scaler reduced its likely loss scale from 2,048 to 128; scale 128 was never attempted.

This incident is classified as premature dynamic-loss-scale calibration termination, not a repeat of the GDN-2 forward non-finite failure. The last successful loss, gradient, throughput, and memory telemetry remained ordinary for this run.

The repository now derives an execution-time retry allowance from the restored GradScaler scale and backoff factor. The configured retry count remains serialized and acts as a minimum, preserving checkpoint identity. A block may retry until it receives a final attempt at loss scale 1.0. If gradients remain non-finite there, the trainer still fails closed with block and scale diagnostics. Non-finite forward loss fails immediately because backward loss scaling cannot repair it.

The next action is to pull current `main` and rerun the normal Kaggle entry point. The launcher reads and verifies the actual remote checkpoint pointer. The expected latest durable boundary is step 1,250 because verified publication occurs every 250 successful updates, but pointer verification remains authoritative.

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
adaptive GDN decay-span limit: 60
configured minimum FP16 overflow retries: 3
adaptive FP16 calibration floor: loss scale 1.0
validation cadence: 250 updates
local checkpoint cadence: 250 updates
verified remote publication cadence: 250 updates
repository default session cap: none within the finite plan
W&B run ID: 20m-100m-data-004
pinned recovery worktree: 8e3cd9cb149facc5fa28e8108a70304c1f8c1c15
```

W&B reconnects to the same fixed run and does not erase failed or replayed history tails. Model/data resume remains exact from the verified checkpoint.

This file does not guess live progress after launch. Read the active W&B run and verified remote checkpoint pointer for exact state.

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
- Use adaptive numerical GDN subchunking rather than globally shrinking the configured chunk or clamping decay.
- Let FP16 loss scaling calibrate to scale 1.0 before failing an otherwise atomic block.
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

The post-pretraining prompt suite now supports `--max-new-tokens` as a global cap on each case's native generation budget and `--trace-top-tokens` for per-step raw next-token probability inspection. The canonical short diagnostic uses greedy decoding, a 32-token cap, and a top-5 raw-token trace; the trace is printed and retained in JSON without changing the model's decoding distribution.

The ordinary held-out training validation path uses `torch.inference_mode()` and a dedicated one-sequence microbatch to prevent full-vocabulary evaluation OOM on the T4.

## Current source of truth

- Experiment procedure: [`../runbooks/20m_100m_runbook.md`](../runbooks/20m_100m_runbook.md)
- FP16 incident evidence: [`../evidence/20m_100m/fp16_overflow_step_1497_2026-08-06.md`](../evidence/20m_100m/fp16_overflow_step_1497_2026-08-06.md)
- FP16 calibration decision: [`../decisions/0006-calibrate-fp16-loss-scale-before-failing-block.md`](../decisions/0006-calibrate-fp16-loss-scale-before-failing-block.md)
- FP16 execution contract: [`../reference/fp16_overflow_recovery.md`](../reference/fp16_overflow_recovery.md)
- GDN-2 incident evidence: [`../evidence/20m_100m/gdn2_nonfinite_step_1138_2026-08-06.md`](../evidence/20m_100m/gdn2_nonfinite_step_1138_2026-08-06.md)
- Adaptive GDN-2 decision: [`../decisions/0005-adapt-gdn2-chunks-to-decay-span.md`](../decisions/0005-adapt-gdn2-chunks-to-decay-span.md)
- Validation OOM evidence: [`../evidence/20m_100m/validation_oom_step_500_2026-08-06.md`](../evidence/20m_100m/validation_oom_step_500_2026-08-06.md)
- Revised durability decision: [`../decisions/0004-run-100m-in-one-session-with-250-step-durability.md`](../decisions/0004-run-100m-in-one-session-with-250-step-durability.md)
- Evaluation procedure: [`../runbooks/eval_core_v1_runbook.md`](../runbooks/eval_core_v1_runbook.md)
- Prompt-suite procedure: [`../runbooks/post_pretraining_prompt_suite.md`](../runbooks/post_pretraining_prompt_suite.md)
- Model contract: [`../reference/model_architecture.md`](../reference/model_architecture.md)
- GDN-2 execution contract: [`../reference/gdn2_chunkwise_training.md`](../reference/gdn2_chunkwise_training.md)
- Dataset contract: [`../reference/dataset_and_tokenization.md`](../reference/dataset_and_tokenization.md)
- Durable decisions: [`../decisions/README.md`](../decisions/README.md)