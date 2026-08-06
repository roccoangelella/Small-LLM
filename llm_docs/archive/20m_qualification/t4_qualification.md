# T4 Model Qualification

_Last updated: 2026-08-01_

## Decision and test contract

On 2026-08-01 the user approved correcting the T4 qualification harness after investigation showed that the first parity test did not reproduce the real GDN-2 recurrence contract.

The executable entry point is:

```bash
python -m tests.t4_qualification
```

Exact Kaggle commands are documented in `tests/README.md`.

## Why schema version 1 was invalid for parity

The schema-v1 harness used unconstrained Gaussian Q/K vectors and an order-one random initial state. The actual GDN-2 layer L2-normalizes Q and K before the recurrence, starts every independent training record from a zero FP32 state, and carries a nonzero state only during segmented or cache execution.

Those synthetic inputs could make the recurrence explode and amplify harmless evaluation-order differences. The first report remains valid for execution, rough memory, and throughput evidence, but its parity failures are not mathematical evidence.

No model code was changed to correct the test.

## Schema-v2 parity contract

For chunk sizes 16, 32, and 64, and for FP32 and FP16-quantized inputs, the harness compares chunkwise GDN-2 with the tokenwise oracle using:

1. `training_zero_state`: normalized Q/K and a zero FP32 initial state;
2. `bounded_cache_state`: normalized Q/K and a small bounded FP32 carried state.

It checks independently:

- every token output;
- final FP32 recurrent state;
- gradients with respect to Q, K, V, log-decay, erase gate, write gate, and initial state.

Parity runs with CUDA autocast disabled. FP16 parity means FP16-quantized recurrence inputs entering the explicit FP32 recurrence core. Full-model FP16 benchmarking remains separate and runs under real CUDA autocast.

## Corrected T4 result

The schema-v2 report at commit `ecee3cab99d23b0db5311a61b6fdd6274ed5b808` passed all 12 parity cases: every chunk size, precision-input mode, state profile, output comparison, final-state comparison, and named-gradient comparison passed within the configured tolerances.

This clears the suspected chunkwise-versus-tokenwise logic defect for the tested contract.

## Operational mixed-precision result

The approximately-20M model was benchmarked at context 2,048 and microbatch 1.

- FP32 chunks 16, 32, and 64 passed.
- FP16 chunks 16 and 32 passed with finite loss and gradients and no scaler reductions.
- FP16 chunk 64 still failed with non-finite chunkwise values.
- FP16 chunk 32 is the fastest GDN-2 candidate that passed both parity and full-model training, at approximately 1,291 tokens/s.
- Plan B passed at approximately 17,260 tokens/s, about 13.4 times faster in this short ordinary-PyTorch comparison.

Chunk 64's mathematical parity pass plus full-model FP16 failure localizes the problem to mixed-precision execution or numerical range, not the recurrence equations.

## Initialization result

Initializer screening used FP16 chunk 32 at context 256.

- normal initialization passed with decreasing loss, finite gradients, and no scaler reductions;
- Xavier initialization failed with NaN gradients and a scaler reduction on every measured step.

Normal initialization is therefore the current candidate for bounded T4 FP16 smoke work. The final all-scale initialization policy remains formally open pending integrated and repeated evidence.

## Candidate-selection status

The harness recommends:

```text
architecture: gdn2_hybrid
backend: pytorch_chunkwise
chunk size: 32
precision: fp16
status: candidate
```

The report does not automatically change `ModelConfig.gdn_chunk_size=64`. An explicit project decision is required before changing the frozen default. Until then, chunk 32 should be supplied explicitly for bounded T4 FP16 qualification runs.

## Remaining qualification work

The mathematical parity gate is complete. Remaining model-side work is:

1. run integrated schema-v2 trainer and checkpoint tests using GDN-2 chunk 32 and normal initialization;
2. remove or revise the trainer CLI's stale parity-defect safety message before trusted GDN-2 use;
3. measure longer-run FP16 stability rather than only three optimizer steps;
4. investigate the chunk-64 autocast/non-finite path or leave chunk 64 unqualified for FP16;
5. qualify a faster optimized GDN-2 backend because the ordinary-PyTorch path is much slower than Plan B;
6. repeat initialization evidence across seeds and a longer bounded run before formally freezing it.

The approximately-100M substantive run remains blocked by integrated trainer, dataset, checkpoint/resume, stability, and throughput gates—not by recurrent/chunkwise mathematical parity.
