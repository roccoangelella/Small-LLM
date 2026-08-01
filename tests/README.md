# Model and T4 qualification tests

Ordinary repository tests remain CPU-friendly and run through `unittest`. The T4 qualification harness is a separate hardware acceptance command because it requires CUDA, performs timed training steps, and writes benchmark results.

## What the T4 harness checks

`tests/t4_qualification.py` performs four related checks:

1. compares the PyTorch chunkwise GDN-2 path with the tokenwise recurrent oracle for token outputs, final recurrent state, and gradients in FP32 and FP16;
2. benchmarks full approximately-20M smoke-model training steps at context 2,048 for chunk sizes 16, 32, and 64;
3. records loss, gradient norm, FP16 scaler reductions, peak allocated/reserved GPU memory, step time, and tokens per second;
4. runs a short FP16 normal-versus-Xavier initialization probe and can optionally benchmark the Plan-B `SWA-512` fallback.

The recurrence parity case is intentionally small. Running the serial recurrent oracle over the full model at context 2,048 would measure Python-loop overhead rather than the intended training path. The 2,048-token benchmark therefore uses the full smoke model with the chunkwise backend.

## Kaggle preparation

Enable a GPU accelerator in the Kaggle notebook and ensure the repository is the current working directory. Kaggle images normally include PyTorch, so the repository can be run directly without installing the project package.

Confirm CUDA and the assigned GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA')"
```

Run the CPU-only control tests first:

```bash
python -m unittest tests.test_t4_qualification -v
```

## Quick T4 smoke command

This checks one chunk size with one measured step and is useful for validating the notebook environment:

```bash
python -m tests.t4_qualification \
  --require-t4 \
  --chunk-sizes 64 \
  --precisions fp32 fp16 \
  --warmup-steps 0 \
  --measure-steps 1 \
  --initialization-steps 1 \
  --output /kaggle/working/t4_qualification_quick.json
```

## Full qualification command

Use this command for the project decision report:

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
  --output /kaggle/working/t4_qualification.json
```

The process exits with:

- `0` when every requested parity case passes and at least one parity-qualified training candidate completes;
- `1` when parity fails or no viable candidate completes;
- `2` for an environment or command error, such as CUDA being unavailable or `--require-t4` selecting another GPU.

## Reading the report

The JSON report contains:

- `environment`: Git commit, Python/PyTorch/CUDA versions, GPU name, compute capability, and memory;
- `parity`: FP32/FP16 output, state, and gradient qualification for each chunk size;
- `benchmarks`: full smoke-model loss, gradient, overflow, memory, timing, and throughput results;
- `initialization_probe`: short normal-versus-Xavier FP16 results;
- `recommendation`: the fastest viable chunk candidate, or Plan B only when no GDN-2 candidate qualifies.

A recommendation is evidence for review, not an automatic configuration change. Repeat the full run before changing the frozen default chunk size or initialization, and commit the reviewed report or its summarized measurements separately.

## Other useful commands

Run the complete ordinary test suite:

```bash
python -m unittest discover -v
```

Skip the initialization probe when only recurrence and throughput are needed:

```bash
python -m tests.t4_qualification --require-t4 --skip-initialization-probe --output /kaggle/working/t4_only.json
```
