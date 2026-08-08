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

## FLA v0.5.1 training status — BLOCKED

Trainer-realistic AMP testing found a decay-dependent backward failure in FLA v0.5.1:

```text
passing forced constant g: [-0.25, -0.5]
failing forced constant g: [-0.75, -1.0, -1.25, -1.5, -2.0, -3.0, -4.0, -5.0, -6.0]
first failing tested point: g=-0.75
```

Real `step-00004000` telemetry then reported:

```text
any_individual_g_le_minus_0.75: True
any_64tok_mean_g_le_minus_0.75: True
real checkpoint overlaps the tested FLA failure region
```

Therefore FLA v0.5.1 `chunk_gdn2` is not qualified for resumed training of this trajectory.

## Latest upstream correction — test FLA v0.5.2 next

FLA v0.5.2 was released on 2026-07-27 and is newer than the v0.5.1 build currently integrated in Small-LLM. A dedicated T4 qualification probe now forces v0.5.2 and repeats the exact trainer-AMP decay sweep:

```bash
python kaggle/run_gdn2_fla_052_amp_decay_sweep.py
```

Do not change the production launcher to v0.5.2 until that test passes.

Upstream GDN-2 `fused_recurrent_gdn2` is inference-only/forward-only and explicitly does not track gradients, so it is not a pretraining fallback. FlashQLA in v0.5.2 targets the older gated-delta-rule path and requires SM90/SM100-class hardware, so it is not applicable to the Tesla T4 active run.

Detailed backend source of truth: [`gdn2_fla_qualification.md`](gdn2_fla_qualification.md).

## Immediate next gate

Run the v0.5.2 decay sweep.

- If v0.5.2 still fails in the real checkpoint's decay region, reject released FLA chunk training for this trajectory. Then choose between continuing with the adaptive PyTorch backend or engineering/qualifying another exact differentiable training kernel.
- If v0.5.2 passes through the entire synthetic range to `g=-6`, run a direct real step-4000 forward/backward finite-gradient parity test before production integration.

Do not clip/bound learned decay merely to make a kernel pass.

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
