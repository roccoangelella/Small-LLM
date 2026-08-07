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

The next action remains completion of the normal 100M Kaggle entry point followed by the frozen final evaluation bundle. The launcher reads and verifies the actual remote checkpoint pointer rather than relying on this file for live progress.

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

## Authorized next experiment

After the 100M run completes and its frozen evaluation bundle is preserved, the same approximately-20M-parameter model receives one final fresh data-scaling probe over a separately identified approximately-500M-token finite dataset.

The 500M probe is not a continuation of the 100M checkpoint. It starts from the same seed-17 initialization policy and receives a new one-pass WSD schedule derived from the completed 500M manifest. The nominal 500M point is approximately 24.2 accepted source tokens per learned parameter and is intended to characterize the practical limit of the smoke-scale model before the project moves its main pretraining work to larger parameter counts.

Dataset preparation is already authorized and may run on the VPS while the 100M tail finishes. No whole Nemotron-ClimbMix download is required: the existing pinned deterministic HTTP byte-range production path builds the fixed cache directly. Training must wait for the 100M final evaluation and must consume the fully verified private Kaggle dataset rather than a live source stream.

```text
profile: 20m-500m-data-scaling-v1
dataset run ID: 20m-500m-dataset-001
target accepted source tokens: 500,000,000
minimum accepted source tokens: 450,000,000
hard maximum accepted source tokens: 550,000,000
producer durable checkpoint cadence: 20,000,000 source tokens
context: 2,048
sequences per optimizer block: 16
target shard size: 8 MiB
remote durability: required
fresh initialization seed: 17
training durability cadence: 250 optimizer updates
W&B run ID: 20m-500m-data-001
pinned 500M training worktree: 7c726ab51e4f3ed221d164e2596816da6d54c5cc
```

The one-command VPS entry point is `bash kaggle/build_and_push_500m.sh`. The one-command Kaggle entry point, after the 100M final evaluation is complete, is `python kaggle/run_20m_500m.py`.

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
- Complete and evaluate the current 100M run before launching the fresh 500M training run.
- Use one fresh approximately-500M-token run as the final data-scaling characterization of the 20M smoke model; do not spend 1B tokens on it by default.
- Do not continue the 100M final checkpoint into the 500M schedule; use fresh seed-17 initialization and a separately derived finite WSD plan.
- Do not run the matched all-attention or other mixer baseline yet.
- Revisit architecture comparisons when larger model versions are reached.
- Use adaptive numerical GDN subchunking rather than globally shrinking the configured chunk or clamping decay.
- Let FP16 loss scaling calibrate to scale 1.0 before failing an otherwise atomic block.
- Use the permanent stratified `eval_core_v1` fast/full suites and retain the existing prompt answers in the unified evaluator.
- Attempt the complete remaining finite one-pass schedule in one Kaggle invocation by default; exact verified checkpoints handle platform interruption.
- Validate, checkpoint locally, and publish a verified remote checkpoint every 250 successful updates.
- Retain both free-generation diagnostics and a deterministic teacher-forced held-out confidence/rank diagnostic so later token/model scales can be compared on semantic uncertainty versus confident errors.

## Evaluation state

The repository contains:

```text
small-llm-eval-data build|verify
small-llm-eval fast|full
```

The code, manifest contract, streaming metrics, prompt integration, and offline tests are implemented. The production `eval_core_v1` corpus still needs to be built and its fast/full runtime measured on the T4 before it becomes an accepted evaluation artifact.

The post-pretraining model-output suite supports `--max-new-tokens` as a global cap on each case's native generation budget and `--trace-top-tokens` for per-step raw next-token probability inspection. The canonical short diagnostic uses greedy decoding, a 32-token cap, and a top-5 raw-token trace; the trace is printed and retained in JSON without changing the model's decoding distribution.

The suite now also supports `--teacher-forced-validation`. This mode identity-matches the local validation dataset to the verified checkpoint through `drive_manifest.json`, evaluates the first 4,096 active held-out next-token targets in deterministic order, and records true-token probability/rank, raw top-1/top-5 probabilities, top-5 mass, entropy, sampled loss/perplexity, top-k hit rates, and confidently-wrong rates. It processes one sequence at a time and computes distribution metrics in 256-position chunks to keep full-vocabulary diagnostics bounded on the T4.

The ordinary held-out training validation path uses `torch.inference_mode()` and a dedicated one-sequence microbatch to prevent full-vocabulary evaluation OOM on the T4.

## Current source of truth

- Current 100M experiment procedure: [`../runbooks/20m_100m_runbook.md`](../runbooks/20m_100m_runbook.md)
- Authorized 500M final-probe procedure: [`../runbooks/20m_500m_runbook.md`](../runbooks/20m_500m_runbook.md)
- 500M final-probe decision: [`../decisions/0008-run-500m-final-20m-data-scaling-probe.md`](../decisions/0008-run-500m-final-20m-data-scaling-probe.md)
- FP16 incident evidence: [`../evidence/20m_100m/fp16_overflow_step_1497_2026-08-06.md`](../evidence/20m_100m/fp16_overflow_step_1497_2026-08-06.md)
- FP16 calibration decision: [`../decisions/0006-calibrate-fp16-loss-scale-before-failing-block.md`](../decisions/0006-calibrate-fp16-loss-scale-before-failing-block.md)
- FP16 execution contract: [`../reference/fp16_overflow_recovery.md`](../reference/fp16_overflow_recovery.md)
- GDN-2 incident evidence: [`../evidence/20m_100m/gdn2_nonfinite_step_1138_2026-08-06.md`](../evidence/20m_100m/gdn2_nonfinite_step_1138_2026-08-06.md)
- Adaptive GDN-2 decision: [`../decisions/0005-adapt-gdn2-chunks-to-decay-span.md`](../decisions/0005-adapt-gdn2-chunks-to-decay-span.md)
- Validation OOM evidence: [`../evidence/20m_100m/validation_oom_step_500_2026-08-06.md`](../evidence/20m_100m/validation_oom_step_500_2026-08-06.md)
- Revised durability decision: [`../decisions/0004-run-100m-in-one-session-with-250-step-durability.md`](../decisions/0004-run-100m-in-one-session-with-250-step-durability.md)
- Evaluation procedure: [`../runbooks/eval_core_v1_runbook.md`](../runbooks/eval_core_v1_runbook.md)
- Model-output procedure: [`../runbooks/post_pretraining_prompt_suite.md`](../runbooks/post_pretraining_prompt_suite.md)
- Model contract: [`../reference/model_architecture.md`](../reference/model_architecture.md)
- GDN-2 execution contract: [`../reference/gdn2_chunkwise_training.md`](../reference/gdn2_chunkwise_training.md)
- Dataset contract: [`../reference/dataset_and_tokenization.md`](../reference/dataset_and_tokenization.md)
- Durable decisions: [`../decisions/README.md`](../decisions/README.md)
