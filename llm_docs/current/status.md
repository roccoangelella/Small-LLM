---
status: current
last_reviewed: 2026-08-08
---

# Current project status

## Completed 20M / 100M pretraining experiment

The approximately-20M-parameter GDN-2 hybrid completed the fixed approximately-100M-token finite pretraining schedule.

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

Validation improved at every recorded checkpoint from update 250 through update 3,053. The run therefore did not show data saturation of the 20M model.

The exact WSD telemetry reached peak LR `3e-4` at update 153, stayed at peak through update 2,442, began decay at update 2,443, and ended at `3e-5`.

### Numerical stability

The accepted trajectory recorded nine FP16 overflow events. The formerly fatal block succeeded at update 1,498 after four retries, calibrating the loss scale to `128`; no later overflow was recorded through completion.

Across the 3,041 unique successful updates with primary W&B telemetry, median gradient norm was `0.9718`, p95 `1.5728`, maximum `17.8926`; `1,236 / 3,041` logged successful updates were clipped (`40.6%`).

### Resume correctness

Across six Kaggle sessions, 630 successful global updates were replayed after resume and every directly overlapping replay had identical logged loss and gradient norm. This established the same-implementation exact restore/replay contract used before the FLA migration.

### Throughput collapse and resolved diagnosis

Data loading was not the bottleneck:

```text
median data-wait: 4.23 ms
p95 data-wait: 12.25 ms
maximum data-wait: 57.04 ms
maximum reserved VRAM: 9.127 GiB
```

Training throughput fell from about `3,830 target tok/s` early to about `445 target tok/s` late, approximately `8.6x`; validation slowed by almost exactly the same factor.

The original hypothesis was that stronger learned GDN-2 decay caused the correctness-first adaptive PyTorch backend to repeatedly subdivide chunks and synchronize with Python. Controlled FLA tests on the same Tesla T4 now strongly support that diagnosis:

```text
FLA forward correctness: pass
FLA backward correctness: pass
FLA speedup vs adaptive, normal forward: 20.830x
FLA speedup vs adaptive, strong-decay forward: 162.541x
FLA speedup vs adaptive, strong-decay forward+backward: 135.441x
adaptive strong-decay forward retention: 0.086x
FLA strong-decay forward retention: 0.671x
```

The full Small-LLM integrated layer probe also passed:

```text
layer_forward_backward_parity: True
checkpoint_parity: None
INTEGRATION QUALIFIED for checkpoint evaluation
```

Therefore decay clipping/bounding is no longer the preferred runtime fix. FLA is now the accepted CUDA recurrence implementation for checkpoint evaluation and the active 500M resume path.

Detailed current backend state: [`gdn2_fla_qualification.md`](gdn2_fla_qualification.md).

## Active 20M / 500M experiment

The 500M experiment remains its own fresh seed-17 trajectory; it is not initialized from the 100M final checkpoint. It already has its own verified checkpoint chain and may now resume that chain using the FLA execution backend.

```text
profile: 20m-500m-data-scaling-v1
dataset run ID: 20m-500m-dataset-001
target accepted source tokens: 500,000,000
minimum accepted source tokens: 450,000,000
hard maximum accepted source tokens: 550,000,000
context: 2,048
training microbatch: 4
saved/configured gdn_chunk_size: 32
CUDA FLA runtime chunk: 64
held-out validation cadence: 250 successful updates
local checkpoint cadence: 250 successful updates
verified remote publication cadence: 250 successful updates
W&B run ID: 20m-500m-data-001
pinned training worktree: a1471472ca9b5d07f70c844460acffe5c96c5200
```

The historical `gdn_chunk_size=32` model configuration is deliberately preserved because trainer checkpoints compare model configuration strictly. On CUDA, the integrated backend executes the same recurrence with FLA's fixed 64-token kernel internally; the adaptive/reference fallback retains the configured chunk size 32.

The one-click launcher has been repinned to the FLA-integrated implementation. Running:

```bash
python kaggle/run_20m_500m.py
```

continues to restore the latest verified remote 500M checkpoint and resumes its model, optimizer, scheduler, scaler, RNG, data cursor, consumed-token position, and WSD position. No checkpoint tensor conversion is required because FLA adds no learned parameters or state-dict keys.

This resumed path is an explicit implementation migration. Future updates are not expected to be bitwise identical to a hypothetical continuation with the old backend because floating-point operation ordering changes.

A later completely fresh 500M run from update 1 using FLA remains a separate scientific decision if a clean single-backend reference trajectory is desired.

## Historical 10M anchor

```text
accepted source tokens: 10,000,662
optimizer updates: 306
final validation loss: 6.136690
final validation perplexity: 462.520157
FP16 overflow events: 0
```

The 10M result is a historical anchor, not a strict same-validation-block comparison with the 100M result.

## Current frozen/accepted decisions affecting work

- The active 500M trajectory may resume its latest verified checkpoint with FLA GDN-2 CUDA execution.
- Preserve the checkpoint's historical `gdn_chunk_size=32`; do not rewrite model configuration to 64.
- FLA internally uses its fixed 64-token GDN-2 kernel on CUDA.
- Do not clip/bound learned GDN-2 decay based on the old adaptive-backend slowdown.
- Keep the adaptive PyTorch backend as the correctness/reference fallback.
- Start/resume the 500M experiment at microbatch 4; no inherited startup microbatch probes.
- Validate, checkpoint locally, and publish a verified remote checkpoint every 250 successful 500M updates.
- Let FP16 loss scaling calibrate down to scale 1.0 before failing an otherwise atomic block.
- Do not run the matched all-attention baseline at the 20M smoke scale; revisit architecture comparisons at larger sizes.
- Preserve `eval_core_v1` plus free-generation and teacher-forced confidence/rank diagnostics.

## Evaluation state

The completed 100M W&B export establishes the held-out training-validation trajectory. Frozen post-pretraining capability evaluation remains separately preserved/compared through:

```text
eval_core_v1 fast/full
free-generation prompt diagnostics
teacher-forced held-out confidence/rank diagnostics
```

## Current source of truth

- Final 100M W&B evidence: [`../evidence/20m_100m/100m_wandb_final_result_2026-08-07.md`](../evidence/20m_100m/100m_wandb_final_result_2026-08-07.md)
- FLA backend state: [`gdn2_fla_qualification.md`](gdn2_fla_qualification.md)
- Standalone FLA evidence: [`../evidence/gdn2_fla_t4_full_probe_2026-08-08.md`](../evidence/gdn2_fla_t4_full_probe_2026-08-08.md)
- Integrated-layer evidence: [`../evidence/gdn2_fla_layer_integration_2026-08-08.md`](../evidence/gdn2_fla_layer_integration_2026-08-08.md)
- FLA integration decision: [`../decisions/0018-integrate-fla-gdn2-as-checkpoint-compatible-cuda-backend.md`](../decisions/0018-integrate-fla-gdn2-as-checkpoint-compatible-cuda-backend.md)
- 500M FLA resume decision: [`../decisions/0019-resume-500m-checkpoint-with-fla-gdn2-execution.md`](../decisions/0019-resume-500m-checkpoint-with-fla-gdn2-execution.md)
- 500M procedure: [`../runbooks/20m_500m_runbook.md`](../runbooks/20m_500m_runbook.md)
- 500M experiment decision: [`../decisions/0008-run-500m-final-20m-data-scaling-probe.md`](../decisions/0008-run-500m-final-20m-data-scaling-probe.md)
- 500M microbatch/durability decision: [`../decisions/0009-start-500m-at-microbatch-4-with-250-step-durability.md`](../decisions/0009-start-500m-at-microbatch-4-with-250-step-durability.md)
- FP16 calibration decision: [`../decisions/0006-calibrate-fp16-loss-scale-before-failing-block.md`](../decisions/0006-calibrate-fp16-loss-scale-before-failing-block.md)
- Historical adaptive GDN decision: [`../decisions/0005-adapt-gdn2-chunks-to-decay-span.md`](../decisions/0005-adapt-gdn2-chunks-to-decay-span.md)
- GDN-2 execution reference: [`../reference/gdn2_chunkwise_training.md`](../reference/gdn2_chunkwise_training.md)
- Evaluation procedure: [`../runbooks/eval_core_v1_runbook.md`](../runbooks/eval_core_v1_runbook.md)
- Durable decisions: [`../decisions/README.md`](../decisions/README.md)
