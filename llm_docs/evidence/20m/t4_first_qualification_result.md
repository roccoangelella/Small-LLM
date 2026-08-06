# T4 GDN-2 Qualification Results

_Last updated: 2026-08-01_

## Corrected project conclusion

The corrected schema-v2 Kaggle run establishes that the PyTorch chunkwise GDN-2 implementation matches the tokenwise recurrent oracle on the NVIDIA Tesla T4 for the tested model-relevant inputs.

All 12 parity cases passed:

- chunk sizes 16, 32, and 64;
- FP32 inputs and FP16-quantized inputs entering the explicit FP32 recurrence core;
- zero-state independent-training records and bounded carried-state/cache records;
- every token output, final recurrent state, and gradients with respect to Q, K, V, log-decay, erase gate, write gate, and initial state.

The previously suspected recurrent/chunkwise algebra defect is therefore cleared for this qualification scope. The schema-v1 failures were caused by an invalid stress input profile with unnormalized Gaussian Q/K and an order-one random state, not by demonstrated disagreement between the two recurrence formulations.

## Full-model operational results

The approximately-20M smoke model was benchmarked at context 2,048, microbatch 1, with three measured optimizer steps after one warmup step.

### GDN-2 results

| Precision | Chunk | Status | Tokens/s | Peak allocated MiB |
|---|---:|---|---:|---:|
| FP32 | 16 | pass | 699.0 | 2,776.9 |
| FP32 | 32 | pass | 1,378.3 | 2,760.0 |
| FP32 | 64 | pass | 2,581.5 | 2,764.5 |
| FP16 | 16 | pass | 648.8 | 2,386.4 |
| FP16 | 32 | pass | 1,291.3 | 2,347.2 |
| FP16 | 64 | fail: non-finite chunkwise values | — | — |

Successful runs had decreasing loss, finite gradient norms, and no FP16 scaler reductions.

Chunk size 64 is mathematically parity-qualified but is not operationally qualified under the current full-model FP16 autocast path. This isolates the remaining chunk-64 problem to mixed-precision execution or numerical range rather than the chunkwise equations themselves.

Chunk size 32 is the current T4 FP16 GDN-2 candidate because it passed every parity profile and the full-model FP16 benchmark. This is evidence for review, not yet a frozen replacement for `ModelConfig.gdn_chunk_size=64`.

## Plan-B comparison

The Plan-B `SWA-512` hybrid passed FP16 at approximately 17,260 tokens/s and about 2,619 MiB peak allocated memory.

In this short ordinary-PyTorch benchmark, Plan B was approximately 13.4 times faster than the qualified GDN-2 FP16 chunk-32 path. GDN-2 correctness is now established, but throughput remains a major operational concern unless a substantially faster backend is qualified.

## Initialization probe

The initializer screen used the qualified FP16 chunk size 32 at context 256 for three optimizer steps.

- GPT-style normal initialization passed: loss decreased from approximately 10.87 to 9.71, gradient norms remained finite, and no overflow occurred.
- Xavier initialization failed: gradients were NaN on all measured steps, the scaler reduced on every step, and loss did not improve.

The current evidence therefore favors normal initialization for T4 FP16 smoke training and rejects the present Xavier recipe for that setup. This short probe is strong screening evidence but does not by itself freeze the final initialization policy for every scale or precision.

## Current authorization boundary

The corrected parity gate is passed. The next GDN-2 work is no longer to repair an alleged sequential/chunkwise logic mismatch; it is to:

1. use chunk size 32 for bounded T4 FP16 integration experiments unless an explicit decision changes the default;
2. investigate or avoid the chunk-64 FP16 non-finite path;
3. qualify trainer, dataset consumption, checkpoint/resume, and longer-run numerical stability with GDN-2;
4. evaluate an optimized GDN-2 backend because the ordinary-PyTorch path is much slower than Plan B;
5. use normal initialization for the next bounded FP16 smoke experiment while keeping final initialization formally open.

The approximately-100M pretraining run remains unauthorized until the integrated smoke-training and operational gates pass.

## Report identity

- report schema: 2;
- report commit: `ecee3cab99d23b0db5311a61b6fdd6274ed5b808`;
- device: NVIDIA Tesla T4, compute capability 7.5;
- PyTorch: 2.10.0 with CUDA 12.8;
- report timestamp: 2026-08-01T08:41:14Z.
