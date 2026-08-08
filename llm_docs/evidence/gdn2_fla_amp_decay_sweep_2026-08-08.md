# FLA GDN-2 trainer-AMP decay sweep — 2026-08-08

## Purpose

After normal-decay AMP layer gradients passed but forced `log_decay=-6` produced non-finite FLA chunk-backward gradients, sweep progressively stronger constant decay values under the actual trainer precision contract before deciding whether the failure overlaps the real step-4000 model.

Contract:

- Tesla T4;
- FP32 model parameters;
- CUDA FP16 autocast;
- existing Small-LLM GDN-2 layer;
- saved/checkpoint geometry `gdn_chunk_size=32`;
- FLA v0.5.1 runtime chunk 64;
- full layer forward/backward gradient parity against the adaptive reference.

## User-reported result

```text
passing: [-0.25, -0.5]
failing: [-0.75, -1.0, -1.25, -1.5, -2.0, -3.0, -4.0, -5.0, -6.0]
first failing tested point: g=-0.75 (64-token cumulative magnitude 48.0)
```

## Interpretation

The failure boundary is substantially milder than the original `g=-6` stress case. FLA v0.5.1 chunk backward is therefore unsafe to authorize from the synthetic tests alone once 64-token regions approach the tested constant `g=-0.75` regime.

This does not yet prove that the real step-4000 checkpoint is unsafe, because the model produces non-constant, input-dependent log-decay tensors. The next gate is to run forward-only telemetry on the exact step-4000 checkpoint and representative real training data, recording per-layer log-decay quantiles and 64-token cumulative magnitudes/means.

## Decision boundary

- Do not resume 500M FLA chunk training yet.
- Do not infer that clipping/bounding decay is required.
- Measure the real checkpoint before choosing FLA chunk, an exact-recurrence fallback, or any model-semantic change.
