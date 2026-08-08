---
status: current
last_reviewed: 2026-08-08
---

# Current project status

## Completed 20M / 100M pretraining experiment

The approximately-20M-parameter GDN-2 hybrid completed its fixed approximately-100M-token pretraining schedule.

Canonical final evidence:

```text
W&B run ID: 20m-100m-data-004
optimizer updates: 3,053
consumed training target tokens: 100,018,176
final validation loss: 4.252758495143203
final validation perplexity: 70.29906475797992
final checkpoint: step-00003053
```

The run stayed trainable but suffered an approximately 8.6x throughput collapse, from roughly 3,830 target tok/s early to roughly 445 tok/s late. Data wait remained negligible. Controlled backend experiments strongly support the diagnosis that stronger learned GDN-2 decay exposed pathological subdivision and synchronization in the adaptive PyTorch chunk backend.

## Active 20M / 500M experiment

The 500M run is an independent seed-17 trajectory, not a continuation of the completed 100M run.

Current verified trajectory point:

```text
checkpoint: step-00004000
last_consumed_block_id: 3999
next intended update: 4001
W&B run ID: 20m-500m-data-001
context: 2048
microbatch: 4
saved gdn_chunk_size: 32
```

The checkpoint is clean. No FLA migration attempt has committed a successful update 4001.

## Released FLA chunk training status — BLOCKED

Trainer-realistic FP32-master + FP16-autocast testing found decay-dependent backward failures in both released FLA versions tested.

### v0.5.1

```text
passing forced constant g: [-0.25, -0.5]
failing: [-0.75, -1.0, -1.25, -1.5, -2.0, -3.0, -4.0, -5.0, -6.0]
first failing tested point: g=-0.75
```

Real step-4000 telemetry then reported:

```text
any_individual_g_le_minus_0.75: True
any_64tok_mean_g_le_minus_0.75: True
real checkpoint overlaps the tested FLA failure region
```

### v0.5.2

A later repeat forced `fla-core==0.5.2` and still failed:

```text
passing: [-0.25, -0.5, -1.0]
failing: [-0.75, -1.25, -1.5, -2.0, -3.0, -4.0, -5.0, -6.0]
first failing tested point: g=-0.75
VERDICT: v0.5.2 still has a tested trainer-AMP backward failure
```

The v0.5.2 pattern is non-monotonic because `g=-1.0` passed while `g=-0.75` and `g=-1.25` failed. Do not describe the issue as a simple magnitude threshold. The production conclusion is still decisive: released FLA chunk training has backward failures in a decay regime that overlaps the actual step-4000 model.

Therefore neither FLA v0.5.1 nor v0.5.2 `chunk_gdn2` is qualified for resumed training of this trajectory.

Upstream `fused_recurrent_gdn2` is inference-only/forward-only and does not track gradients, so it is not a pretraining fallback. FlashQLA is not applicable to the active Tesla T4 / SM75 GDN-2 setup.

## Immediate production boundary

Do **not** run the active 500M production continuation with the currently integrated FLA chunk backend.

The current launcher had previously been repinned for FLA integration, so it must not be treated as safe to resume until a new backend decision is implemented.

Safe accepted state remains:

```text
step-00004000
last consumed block 3999
next update 4001
```

## Next decision

Choose between:

1. restoring the adaptive PyTorch backend for production and resuming from step 4000 despite expected late-run slowness; or
2. engineering/researching and qualifying another exact differentiable GDN-2 training kernel that supports Tesla T4 / SM75 and the real step-4000 decay distribution.

Do not clip/bound learned decay merely to make a kernel pass. That would change model/training semantics and requires a separate explicit decision.

## Frozen/accepted decisions still in force

- Preserve checkpoint/model config `gdn_chunk_size=32`.
- The latest accepted 500M checkpoint is `step-00004000`.
- Keep the adaptive PyTorch backend as the correctness/reference implementation.
- Do not clip/bound GDN-2 decay solely because of backend runtime behavior.
- Start/resume the 500M experiment at microbatch 4.
- Validate/checkpoint/publish every 250 successful 500M updates.
- Let FP16 loss scaling calibrate down to scale 1.0 before failing an otherwise atomic block.
- Preserve `eval_core_v1` plus free-generation and teacher-forced confidence/rank diagnostics.
- Fresh FLA-from-update-1 training remains unauthorized while released FLA backward is unqualified.

## Current source of truth

- Consolidated long-chat handoff: [`gdn2_fla_investigation_handoff.md`](gdn2_fla_investigation_handoff.md)
- Detailed FLA qualification state: [`gdn2_fla_qualification.md`](gdn2_fla_qualification.md)
- v0.5.1 decay sweep: [`../evidence/gdn2_fla_amp_decay_sweep_2026-08-08.md`](../evidence/gdn2_fla_amp_decay_sweep_2026-08-08.md)
- Real step-4000 decay overlap: [`../evidence/gdn2_step4000_real_decay_overlap_2026-08-08.md`](../evidence/gdn2_step4000_real_decay_overlap_2026-08-08.md)
- v0.5.2 decay sweep: [`../evidence/gdn2_fla_052_amp_decay_sweep_2026-08-08.md`](../evidence/gdn2_fla_052_amp_decay_sweep_2026-08-08.md)
- Final 100M evidence: [`../evidence/20m_100m/100m_wandb_final_result_2026-08-07.md`](../evidence/20m_100m/100m_wandb_final_result_2026-08-07.md)
- Durable decisions: [`../decisions/README.md`](../decisions/README.md)
