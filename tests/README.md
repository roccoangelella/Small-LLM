# Model and T4 qualification tests

Ordinary repository tests remain CPU-friendly and run through `unittest`. The T4 qualification harness is a separate hardware-acceptance command because it requires CUDA, performs timed optimizer steps, and writes a benchmark report.

## Corrected parity contract

The original schema-v1 harness generated unconstrained Gaussian Q/K tensors and an order-one random recurrent state. That did not match the real GDN-2 layer, which L2-normalizes Q/K and starts independent training records from a zero FP32 state. Those synthetic inputs could make the recurrence explode and invalidated the first report's parity conclusion.

Schema version 2 fixes the test rather than changing the model:

- Q and K are L2-normalized before the recurrence;
- the `training_zero_state` profile starts from an all-zero FP32 state;
- the `bounded_cache_state` profile uses a small FP32 state to test carried-state behavior;
- parity is evaluated outside CUDA autocast, so FP16 parity means FP16-quantized inputs entering the recurrence's explicit FP32 core;
- full-model FP16 benchmarks still run under CUDA autocast and remain the operational stability test;
- initialization screening uses the fastest FP16 chunk that first passes every requested parity profile and the full-model benchmark, rather than always forcing chunk 64.

The old schema-v1 report remains useful for execution, memory, and throughput evidence. Its parity failures must not be treated as proof of a mathematical GDN-2 defect.

## What the harness checks

`tests/t4_qualification.py` performs four checks:

1. compares chunkwise GDN-2 with the tokenwise recurrent oracle for outputs, final state, and every recurrence-input gradient in FP32 and FP16-input modes;
2. runs both zero-state training parity and bounded carried-state parity for chunk sizes 16, 32, and 64;
3. benchmarks the approximately-20M smoke model at context 2,048, recording loss, gradient norm, FP16 scaler reductions, memory, step time, and tokens per second;
4. screens normal versus Xavier initialization on a parity-qualified FP16 chunk and can optionally benchmark Plan B.

The recurrence cases intentionally remain small. Running the serial Python oracle across the full 2,048-token model would benchmark Python-loop overhead rather than the intended training path.

## Kaggle preparation

Enable a GPU accelerator in Kaggle and make the repository the current working directory.

Confirm the assigned device:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA')"
```

Run the CPU-only control tests first:

```bash
python -m unittest tests.test_t4_qualification -v
```

## Quick corrected T4 check

```bash
python -m tests.t4_qualification \
  --require-t4 \
  --chunk-sizes 32 \
  --precisions fp32 fp16 \
  --warmup-steps 0 \
  --measure-steps 1 \
  --initialization-steps 1 \
  --output /kaggle/working/t4_qualification_v2_quick.json
```

## Full corrected qualification

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

The process exits with:

- `0` when every requested bounded parity case passes and at least one parity-qualified training candidate completes;
- `1` when parity fails or no viable candidate completes;
- `2` for an environment or command error.

## Reading the schema-v2 report

The JSON report contains:

- `parity_contract`: the exact Q/K normalization, state, autocast, and FP16 semantics;
- `parity`: separate output, final-state, and named-gradient comparisons for every profile;
- `benchmarks`: full-model losses, gradients, overflow behavior, memory, and throughput;
- `initialization_probe_chunk_size`: the qualified chunk used for initializer screening, or `null` when the probe was skipped;
- `recommendation`: the fastest viable GDN-2 candidate, or Plan B only when no GDN-2 candidate qualifies.

A recommendation is evidence for review, not an automatic configuration change. Repeat the full run before changing the frozen default chunk size or initialization.

Run the complete ordinary suite with:

```bash
python -m unittest discover -v
```
