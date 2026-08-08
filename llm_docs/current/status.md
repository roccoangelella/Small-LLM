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

The run stayed numerically trainable but suffered an approximately 8.6x throughput collapse, from roughly 3,830 target tok/s early to roughly 445 tok/s late. Data wait remained negligible. Controlled backend experiments strongly support the diagnosis that stronger learned GDN-2 decay exposed pathological subdivision and synchronization in the adaptive PyTorch chunk backend.

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

The checkpoint is clean. All FLA migration attempts failed before a successful update 4001 was committed.

## FLA v0.5.1 training status — BLOCKED

FLA was investigated because its GDN-2 kernels were dramatically faster than the adaptive PyTorch backend on the same Tesla T4. Forward parity and normal-decay AMP backward parity passed.

However, trainer-realistic AMP backward testing found a decay-dependent numerical failure:

```text
passing forced constant g: [-0.25, -0.5]
failing forced constant g: [-0.75, -1.0, -1.25, -1.5, -2.0, -3.0, -4.0, -5.0, -6.0]
first failing tested point: g=-0.75
```

The decisive real-checkpoint telemetry on `step-00004000` then reported:

```text
any_individual_g_le_minus_0.75: True
any_64tok_mean_g_le_minus_0.75: True
real checkpoint overlaps the tested FLA failure region
```

Therefore FLA v0.5.1 `chunk_gdn2` is **not qualified for resumed training of the active 500M trajectory**.

Current backend source of truth: [`gdn2_fla_qualification.md`](gdn2_fla_qualification.md).

## Immediate engineering choices

The project now has two legitimate paths:

1. **Continue the existing 500M trajectory immediately with the previous adaptive PyTorch backend.** This preserves the exact model/checkpoint semantics but accepts the known throughput collapse.
2. **Pause the 500M trajectory while qualifying another exact-recurrence optimized backend.** A fused/recurrent GDN-2 kernel is the next candidate to investigate before considering any learned-decay clamp/bound.

No decision between those two paths has yet been recorded.

Do not resume with FLA chunk backward, and do not clip/bound learned decay merely to make that kernel work.

## Frozen/accepted decisions still in force

- Preserve the checkpoint's historical `gdn_chunk_size=32` model configuration.
- The latest accepted 500M checkpoint is `step-00004000`.
- Keep the adaptive PyTorch backend as the correctness/reference implementation.
- Do not clip/bound GDN-2 decay solely because of backend runtime behavior.
- Start/resume the 500M experiment at microbatch 4.
- Validate/checkpoint/publish every 250 successful 500M updates.
- Let FP16 loss scaling calibrate down to scale 1.0 before failing an otherwise atomic block.
- Preserve `eval_core_v1` plus free-generation and teacher-forced confidence/rank diagnostics.

## Current source of truth

- Final 100M W&B evidence: [`../evidence/20m_100m/100m_wandb_final_result_2026-08-07.md`](../evidence/20m_100m/100m_wandb_final_result_2026-08-07.md)
- FLA backend state: [`gdn2_fla_qualification.md`](gdn2_fla_qualification.md)
- Decay sweep evidence: [`../evidence/gdn2_fla_amp_decay_sweep_2026-08-08.md`](../evidence/gdn2_fla_amp_decay_sweep_2026-08-08.md)
- Real step-4000 overlap evidence: [`../evidence/gdn2_step4000_real_decay_overlap_2026-08-08.md`](../evidence/gdn2_step4000_real_decay_overlap_2026-08-08.md)
- 500M procedure: [`../runbooks/20m_500m_runbook.md`](../runbooks/20m_500m_runbook.md)
- Durable decisions: [`../decisions/README.md`](../decisions/README.md)
