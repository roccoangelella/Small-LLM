---
status: current
last_reviewed: 2026-08-07
---

# Current project status

## Completed 20M / 100M pretraining experiment

The approximately-20M-parameter GDN-2 hybrid has completed the fixed approximately-100M-token finite pretraining schedule.

Canonical final W&B evidence:

```text
W&B run ID: 20m-100m-data-004
W&B state: finished
optimizer updates: 3,053
final consumed training target tokens: 100,018,176
final held-out validation loss: 4.252758495143203
final held-out validation perplexity: 70.29906475797992
final checkpoint: step-00003053
final remote publication: verified, final=true
```

The validation curve improved at every recorded checkpoint from update 250 through update 3,053. It fell from `6.517897 / 677.153 PPL` at update 250 to `4.252758 / 70.299 PPL` at completion. Validation still improved materially in the tail: `4.450810` at update 2,250, `4.395521` at update 2,500, `4.324534` at update 2,750, `4.260713` at update 3,000, and `4.252758` at the final partial block. The 100M result therefore does not show data saturation of the 20M model; marginal gains were diminishing but remained measurable at the end.

The exact WSD telemetry reaches peak LR `3e-4` at update 153, remains at peak through update 2,442, begins decay at update 2,443, and ends at `3e-5`.

### Numerical stability

The completed accepted trajectory records nine FP16 overflow events. One-retry events occur at updates 939, 1,066, 1,083, 1,199, and 1,247. The formerly fatal block succeeds at update 1,498 after four retries, calibrating the loss scale to `128`. No later overflow is recorded through completion; scale 128 remains stable for the rest of the run.

Across the 3,041 unique successful updates with primary W&B train telemetry, median gradient norm is `0.9718`, p95 is `1.5728`, and the maximum is `17.8926` at update 1,144. `1,236 / 3,041` logged successful updates are clipped (`40.6%`). The run remains finite and validation continues improving despite common clipping.

### Resume correctness

The W&B run preserves replayed tails across six Kaggle sessions. Exactly 630 successful global updates are logged in both an earlier/discarded tail and the resumed trajectory. Every directly overlapping replay has identical logged loss and gradient norm (`max absolute difference = 0`). This is strong empirical evidence that the same-T4 exact restore/replay contract preserved training numerics.

W&B has one evidence gap: updates `2,473–2,484` are absent entirely from the export. No values are inferred for those 12 updates.

### Throughput is now the main implementation concern

Data loading was not the bottleneck:

```text
median data-wait per logged update: 4.23 ms
p95 data-wait: 12.25 ms
maximum data-wait: 57.04 ms
maximum reserved VRAM: 9.127 GiB
```

Training compute slowed severely as model state evolved. On the accepted session path, representative median throughput fell from about `3,830 target tok/s` over updates 1–1,000 to about `445 target tok/s` over updates 3,001–3,053, an approximately `8.6x` slowdown. Validation time shows almost the same factor, from roughly `27.55 s` at update 1,000 to `236.20 s` at the final checkpoint.

This is strongly consistent with the correctness-first adaptive GDN-2 backend increasingly selecting smaller numerical subchunks as learned decay spans grow, but the current telemetry does not log actual selected subchunk sizes or split counts. Causality is therefore not yet proven. Instrumenting GDN subchunk selection / decay-span triggers is now a priority for interpreting 500M runtime and for later larger-model work.

The canonical detailed report is:

- [`../evidence/20m_100m/100m_wandb_final_result_2026-08-07.md`](../evidence/20m_100m/100m_wandb_final_result_2026-08-07.md)

## Active next experiment — fresh 20M / 500M run

The authorized next experiment remains the final fresh data-scaling characterization of the same approximately-20M-parameter model over the separately identified approximately-500M-token finite dataset.

It is not a continuation of the 100M checkpoint. It starts from seed 17 with its own one-pass WSD plan.

```text
profile: 20m-500m-data-scaling-v1
dataset run ID: 20m-500m-dataset-001
target accepted source tokens: 500,000,000
minimum accepted source tokens: 450,000,000
hard maximum accepted source tokens: 550,000,000
context: 2,048
training microbatch: 4 from the first real update
fresh microbatch probes: skipped by experiment decision
held-out validation cadence: 250 successful updates
local checkpoint cadence: 250 successful updates
verified remote publication cadence: 250 successful updates
W&B run ID: 20m-500m-data-001
pinned training worktree: 01d562ea1845d0dd128a0458e613c9e677b7381d
```

The first 500M Kaggle launch verified the attached dataset successfully but failed closed while deriving the schedule because the inherited launcher dispatched the 500M manifest to `dataset.qualification_100m_report`. The dataset was not at fault and no optimizer update ran. `main` now rewrites that one inherited dispatch to `dataset.qualification_500m_report`; the next operational action is simply to pull current `main` and rerun:

```bash
python kaggle/run_20m_500m.py
```

The 500M run remains scientifically useful because the 100M result is still improving. Its wall-clock cost is now a separate implementation risk because the 100M run exposed severe late-stage GDN compute slowdown.

## Historical 10M anchor

```text
accepted source tokens: 10,000,662
optimizer updates: 306
final validation loss: 6.136690
final validation perplexity: 462.520157
FP16 overflow events: 0
```

The 10M result is a historical anchor, not a strict same-validation-block comparison with the 100M result.

## Frozen decisions affecting current work

- Complete one fresh approximately-500M-token run as the final data-scaling characterization of the 20M smoke model.
- Do not initialize the 500M run from the 100M final checkpoint.
- Start the 500M run directly at microbatch 4; do not replay the inherited microbatch 1-vs-4 probes.
- Validate, checkpoint locally, and publish a verified remote checkpoint every 250 successful 500M updates.
- Keep adaptive numerical GDN subchunking rather than altering/clamping the recurrence for correctness.
- Let FP16 loss scaling calibrate down to scale 1.0 before failing an otherwise atomic block.
- Do not run the matched all-attention baseline at the 20M smoke scale; revisit architecture comparisons at larger model sizes.
- Preserve the permanent `eval_core_v1` fast/full suite plus free-generation and teacher-forced confidence/rank diagnostics.

## Evaluation state

The W&B export establishes completion and the held-out training-validation trajectory, but it does not contain the frozen post-pretraining capability evaluation.

Still separate from this training result:

```text
eval_core_v1 fast/full
free-generation prompt diagnostics
teacher-forced held-out confidence/rank diagnostics
```

These should be preserved for the 100M checkpoint so the later 500M result can be compared on more than perplexity.

## Current source of truth

- Final 100M W&B evidence: [`../evidence/20m_100m/100m_wandb_final_result_2026-08-07.md`](../evidence/20m_100m/100m_wandb_final_result_2026-08-07.md)
- 500M procedure: [`../runbooks/20m_500m_runbook.md`](../runbooks/20m_500m_runbook.md)
- 500M experiment decision: [`../decisions/0008-run-500m-final-20m-data-scaling-probe.md`](../decisions/0008-run-500m-final-20m-data-scaling-probe.md)
- 500M microbatch/durability decision: [`../decisions/0009-start-500m-at-microbatch-4-with-250-step-durability.md`](../decisions/0009-start-500m-at-microbatch-4-with-250-step-durability.md)
- FP16 calibration decision: [`../decisions/0006-calibrate-fp16-loss-scale-before-failing-block.md`](../decisions/0006-calibrate-fp16-loss-scale-before-failing-block.md)
- GDN-2 adaptive execution decision: [`../decisions/0005-adapt-gdn2-chunks-to-decay-span.md`](../decisions/0005-adapt-gdn2-chunks-to-decay-span.md)
- GDN-2 execution contract: [`../reference/gdn2_chunkwise_training.md`](../reference/gdn2_chunkwise_training.md)
- FP16 execution contract: [`../reference/fp16_overflow_recovery.md`](../reference/fp16_overflow_recovery.md)
- Evaluation procedure: [`../runbooks/eval_core_v1_runbook.md`](../runbooks/eval_core_v1_runbook.md)
- Model-output procedure: [`../runbooks/post_pretraining_prompt_suite.md`](../runbooks/post_pretraining_prompt_suite.md)
- Dataset contract: [`../reference/dataset_and_tokenization.md`](../reference/dataset_and_tokenization.md)
- Durable decisions: [`../decisions/README.md`](../decisions/README.md)
