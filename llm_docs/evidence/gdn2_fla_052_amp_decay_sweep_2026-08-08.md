---
status: evidence
recorded: 2026-08-08
---

# FLA v0.5.2 GDN-2 trainer-AMP decay sweep

User-reported Tesla T4 result from `kaggle/run_gdn2_fla_052_amp_decay_sweep.py`.

Environment/contract:

```text
fla-core: 0.5.2
trainer precision contract: FP32 parameters + CUDA FP16 autocast
saved/configured GDN chunk: 32
FLA runtime chunk: 64
```

Reported summary:

```text
passing: [-0.25, -0.5, -1.0]
failing: [-0.75, -1.25, -1.5, -2.0, -3.0, -4.0, -5.0, -6.0]
first failing tested point: g=-0.75
64-token cumulative magnitude at constant g=-0.75: 48.0
VERDICT: v0.5.2 still has a tested trainer-AMP backward failure; do not resume training yet.
```

## Interpretation

This does **not** establish a monotonic failure threshold because `g=-1.0` passed while `g=-0.75` and `g=-1.25` failed. The result therefore indicates a numerical/kernel instability pattern rather than a simple `|g| > threshold` rule.

The important production fact is unchanged: the released FLA v0.5.2 chunk training path still has backward failures under the exact trainer AMP contract in a decay regime that overlaps the real verified step-4000 checkpoint telemetry.

Therefore upgrading from v0.5.1 to v0.5.2 does not currently qualify FLA `chunk_gdn2` for resuming the active 20M/500M trajectory.

## Consequence

- Do not resume update 4001 with FLA v0.5.1 or v0.5.2 chunk GDN-2 training.
- The latest accepted trajectory point remains `step-00004000`, with `last_consumed_block_id=3999`.
- No FLA experiment has committed update 4001.
- FLA fused recurrent is forward-only/inference-only and cannot serve as the training fallback.
- FlashQLA is not applicable to the active Tesla T4 / GDN-2 geometry.
- Next work must choose between continuing with the adaptive PyTorch backend or engineering/qualifying another exact differentiable GDN-2 training kernel. Decay clipping/bounding remains unauthorized as a kernel workaround.
