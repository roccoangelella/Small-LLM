---
status: observed
observed_at: 2026-08-06
experiment: 20m-100m-data-004
---

# GDN-2 non-finite failure after update 1,138

## Observation

The approximately-20M-parameter GDN-2 hybrid completed optimizer update 1,138 of the 3,053-update one-pass plan. The next training forward failed inside `gdn2_chunkwise_reference`:

```text
ValueError: chunkwise GDN-2 produced non-finite values; reduce gdn_chunk_size or use a qualified fused backend
```

Active geometry:

```text
architecture: gdn2_hybrid
context length: 2,048
gdn_chunk_size: 32
GDN key/value heads: 4
GDN key/value dimension: 64
training precision: FP16
GDN recurrence arithmetic: FP32
```

## Immediate telemetry

The final logged updates remained operationally ordinary:

```text
step 1130: loss 4.7486, grad 1.193 clipped
step 1131: loss 4.6918, grad 0.847
step 1132: loss 4.6026, grad 0.948
step 1133: loss 5.0823, grad 1.260 clipped
step 1134: loss 4.7816, grad 0.925
step 1135: loss 4.7587, grad 0.893
step 1136: loss 5.0931, grad 1.103 clipped
step 1137: loss 4.6640, grad 0.907
step 1138: loss 4.6273, grad 0.833
```

Throughput stayed approximately 3.8k tokens/s, VRAM stayed approximately 9.1 GiB, and the cumulative FP16 overflow count remained fixed at three. The evidence therefore does not show model-wide divergence before the failure.

## Root-cause analysis

The correctness-first chunkwise backend computes cumulative log-decay `G` and factors pairwise decay ratios around a midpoint:

```text
left  = exp(G - center)
right = exp(center - G)
```

The true causal ratio can remain finite while either reciprocal factor, or a discarded anti-causal entry in a dense product before triangular masking, exceeds FP32 range. A deterministic regression with 32 tokens and constant `log_decay=-6` reproduces non-finite output in the old path while the tokenwise recurrent oracle remains finite.

Classification: **backend numerical-stability incident**, not established model divergence.

## Repair

The assembled model now uses an adaptive wrapper that keeps the configured maximum chunk size 32 but bisects a proposed chunk when its cumulative log-decay span exceeds 60. A still-non-finite chunk is retried at smaller sizes down to one token. No model parameter, state-dict key, model configuration field, or optimizer route changes.

## Recovery boundary

Verified remote publication occurs every 250 successful updates. The expected latest durable checkpoint is therefore `step-00001000`; the launcher must still read and verify the actual remote pointer rather than assume it.

A corrected rerun must restore the verified checkpoint, preserve the fixed W&B run ID, and cross update 1,138 before the incident is closed.

## Links

- [`../../decisions/0005-adapt-gdn2-chunks-to-decay-span.md`](../../decisions/0005-adapt-gdn2-chunks-to-decay-span.md)
- [`../../reference/gdn2_chunkwise_training.md`](../../reference/gdn2_chunkwise_training.md)
- [`../../runbooks/20m_100m_runbook.md`](../../runbooks/20m_100m_runbook.md)
