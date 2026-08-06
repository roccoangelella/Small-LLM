---
status: accepted
date: 2026-08-06
supersedes: null
---

# 0005 — Adapt GDN-2 chunks to cumulative decay span

## Context and problem statement

The corrected 20M-parameter / approximately-100M-token run reached optimizer update 1,138 and then failed during the next forward pass with:

```text
ValueError: chunkwise GDN-2 produced non-finite values; reduce gdn_chunk_size or use a qualified fused backend
```

The surrounding loss, gradient norm, throughput, memory use, and FP16-overflow counter did not show model-wide divergence. The correctness-first chunkwise backend centers cumulative log-decay and materializes reciprocal exponential factors before dense products and triangular masking. Strong but valid negative decay can therefore overflow an intermediate even when the tokenwise recurrent equations remain finite.

The configured chunk size is 32 for the active run. Restarting unchanged would restore the same deterministic data and RNG cursor and could reproduce the same failure.

## Considered options

- Restart the unchanged run and treat the failure as transient.
- Permanently reduce `gdn_chunk_size` to a smaller fixed value such as 8.
- Keep the configured maximum chunk size but adaptively bisect only numerically unsafe chunks.
- Replace the backend immediately with an upstream fused kernel.

## Decision outcome

Chosen option: **keep `gdn_chunk_size=32` as the maximum and adaptively bisect unsafe chunks**, because it preserves ordinary throughput and checkpoint geometry while removing the known midpoint-factorization overflow mode.

The assembled `SmallLLM` now uses `StableGatedDeltaNet2`, whose `AdaptiveChunkwiseGDN2Backend`:

- measures the cumulative log-decay span for each proposed chunk;
- keeps the ordinary configured chunk when the span is at most 60;
- bisects a larger-span chunk until it has conservative FP32 exponent and reduction headroom;
- retries any still-non-finite chunk at smaller sizes;
- fails closed only if the exact one-token path is still non-finite.

The underlying correctness reference remains available unchanged for direct parity testing.

## Consequences

### Positive

- Existing checkpoints remain load-compatible because model parameters, state-dict keys, model configuration, and optimizer routing do not change.
- Most chunks retain size 32; only numerically dangerous regions pay the smaller-chunk cost.
- The repair preserves the mathematical recurrence rather than clamping decay or silently replacing non-finite values.
- A one-token fallback provides an exact lower bound on execution granularity for finite inputs.

### Negative or limiting

- Adaptive selection introduces device synchronization and variable work when dangerous decay spans occur.
- This remains a correctness-first PyTorch implementation, not a substitute for qualifying a fused T4 backend.
- A model-level non-finite value can still fail at chunk size 1, as it should.

## Validation

- Regression test: a 32-token chunk with constant `log_decay=-6`, which overflows the old midpoint path, must match the recurrent oracle in outputs, final state, and gradients.
- Compatibility test: stable and legacy GDN-2 layers must expose identical checkpoint parameter keys.
- Assembly test: every GDN layer in `SmallLLM` must use the adaptive backend while retaining configured maximum chunk size 32.
- Operational qualification: restore the verified step-1000 checkpoint on the T4 and pass the previous failure boundary beyond update 1,138 before classifying the incident closed.

## Links

- [`../reference/gdn2_chunkwise_training.md`](../reference/gdn2_chunkwise_training.md)
- [`../runbooks/20m_100m_runbook.md`](../runbooks/20m_100m_runbook.md)
- [`../evidence/20m_100m/gdn2_nonfinite_step_1138_2026-08-06.md`](../evidence/20m_100m/gdn2_nonfinite_step_1138_2026-08-06.md)
