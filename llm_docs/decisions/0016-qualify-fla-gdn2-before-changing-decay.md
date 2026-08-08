---
status: accepted
date: 2026-08-08
supersedes: null
---

# 0016 — Qualify FLA GDN-2 before changing learned decay

## Context and problem statement

The completed 20M / 100M run showed a severe late-training throughput collapse while validation continued improving. Project evidence strongly implicates the correctness-first adaptive GDN-2 backend: learned decay spans can force repeated chunk subdivision and Python/GPU synchronization. The upcoming 20M / 500M run has roughly 15.2k planned updates, so changing or clipping the learned decay after only the early part of training could alter the model for most of the experiment.

Flash Linear Attention (FLA) v0.5.1 now exposes an MIT-licensed optimized GDN-2 chunk training kernel whose recurrence matches the project's GDN-2 state update and whose public tests compare the kernel against a recurrent oracle.

## Decision outcome

Before introducing a decay clip/bound or changing GDN-2 learning semantics, run a dedicated CUDA qualification probe for the FLA GDN-2 backend.

The probe must:

- leave the existing Small-LLM training backend unchanged;
- compare FLA outputs and final recurrent state against the project's tokenwise recurrent oracle;
- compare gradients for Q, K, V, log-decay, erase, write, and initial state;
- include the strong `log_decay=-6` regime used by the existing numerical regression test;
- benchmark normal-decay and strong-decay execution against `AdaptiveChunkwiseGDN2Backend` at the 20M GDN geometry;
- run at batch 4 and context 2,048 by default to approximate the active 500M training microbatch;
- report whether the optimized backend preserves throughput when decay becomes strong;
- write a machine-readable JSON result.

The one-click Kaggle entry point is:

```bash
python kaggle/run_gdn2_fla_t4_probe.py
```

The probe pins `flash-linear-attention==0.5.1`. If FLA is absent it may install that package with `--no-deps`, but it must not automatically replace or upgrade the notebook's PyTorch runtime.

## Consequences

- No decay clipping or bounded-gate change is authorized by this ADR.
- No production backend replacement is authorized by this ADR.
- A successful probe makes FLA a candidate for an adapter/integration qualification against the full Small-LLM layer and checkpoint path.
- If the optimized kernel is correct but performs poorly or cannot execute on T4/Turing, moving the experiment to newer training hardware may be preferable to changing model semantics.
- If optimized-kernel qualification fails, bounded decay remains an explicit fallback experiment rather than an assumed fix.

## Links

- [`../../kaggle/run_gdn2_fla_t4_probe.py`](../../kaggle/run_gdn2_fla_t4_probe.py)
- [`0005-adapt-gdn2-chunks-to-decay-span.md`](0005-adapt-gdn2-chunks-to-decay-span.md)
- [`../reference/gdn2_chunkwise_training.md`](../reference/gdn2_chunkwise_training.md)
- [`../evidence/20m_100m/100m_wandb_final_result_2026-08-07.md`](../evidence/20m_100m/100m_wandb_final_result_2026-08-07.md)
