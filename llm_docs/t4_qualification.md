# T4 Model Qualification

_Last updated: 2026-08-01_

## Decision

On 2026-08-01 the user approved correcting the T4 qualification harness after investigation showed that the first parity test did not reproduce the real GDN-2 recurrence contract.

The executable entry point remains:

```bash
python -m tests.t4_qualification
```

Exact Kaggle commands are documented in `tests/README.md`.

## Why the first parity test was invalid

The schema-v1 harness used unconstrained Gaussian Q/K vectors and an order-one random initial state. The actual GDN-2 layer L2-normalizes Q and K before the recurrence, starts every independent training record from a zero FP32 state, and carries a nonzero state only during segmented/cache execution.

With unnormalized keys, the delta update can become strongly expansive. The resulting synthetic recurrence reached enormous magnitudes, so small floating-point evaluation-order differences became large absolute discrepancies. The first report therefore remains valid as evidence that the model can execute on a T4 and as a rough memory/throughput measurement, but its parity failures are not valid evidence of a chunkwise algebra defect.

No model code was changed by this correction.

## Corrected schema-v2 parity contract

For chunk sizes 16, 32, and 64, and for FP32 and FP16-quantized inputs, the harness now compares the chunkwise backend against the tokenwise oracle using two profiles:

1. `training_zero_state`: normalized Q/K and a zero FP32 initial state;
2. `bounded_cache_state`: normalized Q/K and a small bounded FP32 carried state.

The test compares independently:

- every token output;
- final FP32 recurrent state;
- gradients with respect to Q, K, V, log-decay, erase gate, write gate, and initial state.

Parity runs with CUDA autocast explicitly disabled. FP16 parity therefore means FP16-quantized recurrence inputs entering the implementation's explicit FP32 core. This isolates mathematical equivalence from mixed-precision operational behavior.

## Full-model operational benchmark

The approximately-20M smoke model is still benchmarked with:

- context 2,048;
- microbatch 1 by default;
- next-token cross-entropy;
- AdamW;
- real CUDA FP16 autocast and `GradScaler` for FP16;
- configurable warmup and measured steps.

This benchmark remains responsible for detecting real mixed-precision failures such as the first run's FP16 chunk-64 non-finite behavior. It records losses, global gradient norms, scale reductions, peak allocated/reserved memory, step time, throughput, OOMs, and non-finite failures.

## Initialization probe correction

The schema-v1 harness always used the largest requested chunk for normal-versus-Xavier screening. Because chunk 64 failed FP16 operationally, both initializer probes failed before they could compare initialization behavior.

Schema version 2 selects the fastest FP16 chunk that first:

- passes every requested parity precision and both parity profiles;
- completes the full-model FP16 benchmark with finite loss and gradients;
- produces no scaler reduction during the measured window.

When no such chunk exists, the initialization probe is explicitly skipped instead of producing a misleading initializer failure.

## Candidate-selection rule

A GDN-2 chunk is eligible only when all requested bounded parity cases pass and the corresponding full-model benchmark succeeds. Among eligible candidates, the report prefers FP16 and selects the highest measured tokens per second.

Plan B is labelled a `fallback_candidate` only when no GDN-2 candidate meets both correctness and operational gates. The harness never changes `ModelConfig.gdn_chunk_size` or the architecture automatically.

## Required rerun

The corrected report must use a new output path, for example:

```bash
python -m tests.t4_qualification \
  --require-t4 \
  --chunk-sizes 16 32 64 \
  --precisions fp32 fp16 \
  --sequence-length 2048 \
  --batch-size 1 \
  --warmup-steps 1 \
  --measure-steps 3 \
  --include-plan-b \
  --output /kaggle/working/t4_qualification_v2.json
```

Until that schema-v2 report is run and reviewed, the project conclusion is limited to:

- GDN-2 execution feasibility on the T4 is established;
- the first parity failure is reclassified as a harness defect;
- FP16 chunk-64 non-finite behavior remains real operational evidence;
- trusted GDN-2 pretraining remains blocked pending corrected qualification.
