# First T4 GDN-2 Qualification Result

_Last updated: 2026-08-01_

## Project conclusion

The first Kaggle NVIDIA Tesla T4 run established that the current Small LLM GDN-2 model can execute on compute capability 7.5. This remains valid execution-feasibility evidence, not pretraining authorization.

## Valid evidence from the first run

Using the approximately-20M smoke model at context 2,048 and microbatch 1:

- the PyTorch chunkwise GDN-2 hybrid completed FP32 optimizer steps with chunk sizes 16, 32, and 64;
- it completed FP16 optimizer steps with chunk sizes 16 and 32;
- losses decreased, gradients remained finite, and no scaler reductions occurred in those successful short runs;
- peak allocated memory remained below approximately 2.8 GiB;
- FP16 chunk size 64 produced non-finite values and failed;
- Plan B completed successfully and remained the operational fallback;
- the ordinary-PyTorch GDN-2 path was substantially slower than Plan B in this short benchmark.

These observations remain valid because they came from actual full-model optimizer steps.

## Reclassification of the parity result

The first report's recurrent-versus-chunkwise parity conclusion is withdrawn as mathematical evidence.

The schema-v1 harness generated unnormalized Gaussian Q/K vectors and an order-one random initial state. That differs from the real layer, which L2-normalizes Q/K and starts independent training records from a zero FP32 state. The synthetic recurrence could therefore explode, magnifying harmless evaluation-order differences into huge absolute discrepancies and non-finite FP16 values.

The first report showed that the old test inputs were unstable; it did not establish a chunkwise algebra defect.

## Corrected next gate

The schema-v2 harness now requires:

- normalized Q/K;
- a zero-state training profile;
- a bounded carried-state profile;
- separate output, final-state, and named-gradient comparisons;
- parity outside CUDA autocast;
- operational FP16 testing through the full-model autocast benchmark;
- initializer screening only on a parity-qualified, operationally successful FP16 chunk.

After the corrected Kaggle run, the project can distinguish among:

1. a real mathematical parity defect;
2. an FP16/autocast-only implementation problem;
3. a chunk-size-specific stability problem;
4. a performance limitation without a correctness failure.

## Decision boundary

Do not replace GDN-2 solely because of the schema-v1 parity result. Keep GDN-2 as the intended architecture while rerunning the corrected harness. Trusted GDN-2 CLI pretraining remains blocked until schema-v2 qualification passes. Plan B remains available for trainer-plumbing validation and as the operational fallback.
