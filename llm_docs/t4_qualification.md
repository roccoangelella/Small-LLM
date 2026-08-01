# T4 Model Qualification

_Last updated: 2026-08-01_

## Decision

On 2026-08-01 the user approved implementing a reproducible model-hardware qualification harness under `tests/` and running it on a Google Kaggle notebook that exposes an NVIDIA T4.

The executable entry point is:

```bash
python -m tests.t4_qualification
```

The exact Kaggle commands and report interpretation are documented in `tests/README.md`.

## Purpose

The harness answers three separate questions without changing the model architecture:

1. Does the differentiable chunkwise GDN-2 backend still match the readable recurrent oracle in FP32 and FP16?
2. Which tested chunk size is stable and operationally best on the available T4 at the frozen 2,048-token context?
3. Does the current PyTorch GDN-2 path qualify for smoke pretraining, or is the Plan-B `SWA-512` fallback required?

It also supplies the first target-hardware evidence for choosing between the existing normal and Xavier initialization candidates.

## Implemented checks

### Recurrence correctness

For chunk sizes 16, 32, and 64, the harness compares the chunkwise backend against the tokenwise recurrent oracle for:

- every token output;
- final FP32 recurrent state;
- gradients with respect to Q, K, V, log-decay, erase gate, write gate, and initial state;
- FP32 and FP16 input paths;
- multiple chunks and a shorter final chunk.

The default parity sequence length is 129. Correctness is tested on deliberately small recurrence tensors so the serial oracle remains a useful mathematical reference rather than dominating the full hardware benchmark with a Python token loop.

### Full smoke-model benchmark

For every requested chunk size and precision, the harness constructs the frozen approximately-20M smoke model and runs actual optimizer steps with:

- context length 2,048;
- microbatch 1 by default;
- next-token cross-entropy;
- AdamW;
- CUDA FP16 autocast and `torch.amp.GradScaler` for FP16;
- configurable warmup and measured steps.

It records:

- losses and global gradient norms;
- FP16 scale reductions/overflow events;
- mean step time and tokens per second;
- peak allocated and peak reserved CUDA memory;
- OOM and non-finite failures.

### Initialization probe

Unless explicitly skipped, the harness runs short FP16 training probes for both existing initialization candidates:

- GPT-style normal initialization with standard deviation 0.02;
- Xavier-uniform initialization.

Both retain the shared residual scaling, GDN-specific state initialization, RMSNorm initialization, and padded-vocabulary invariants already implemented by `model.initialization`.

The default initialization probe uses a shorter 256-token context and three steps. It is a screening test, not sufficient by itself to freeze initialization. A final choice requires repeatable T4 evidence and review of early loss, gradient norms, and overflow behavior.

### Plan-B reference

With `--include-plan-b`, the same smoke training-step benchmark is run for the parameter-matched `SWA-512` fallback in FP16. Plan B is recommended only when no GDN-2 chunk candidate passes both recurrence parity and finite training.

## Candidate-selection rule

The generated report prefers FP16 because it is the intended T4 training precision. A GDN-2 chunk size is eligible only when:

- its parity checks pass in every requested precision;
- its full smoke-model benchmark completes;
- measured loss and gradient norm remain finite;
- no FP16 scale reduction is observed during the short measured window.

Among eligible candidates, the report identifies the highest measured tokens-per-second result. This is labelled a `candidate`; it does not silently alter `ModelConfig.gdn_chunk_size=64`.

If no GDN-2 candidate qualifies and the optional Plan-B benchmark succeeds, the report labels Plan B a `fallback_candidate`. If neither path completes, the result is `blocked`.

## Full Kaggle command

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

## Qualification boundary

The harness is now implemented, but no T4 result has been recorded yet. Therefore the current chunkwise PyTorch backend remains mathematically qualified on CPU and operationally unqualified on the target GPU until the Kaggle report is run and reviewed.

This command does not yet validate:

- schema-v2 dataset consumption;
- a full trainer or learning-rate schedule;
- joint checkpoint interruption/resume;
- unified generation caching;
- the approximately-100M substantive geometry;
- an upstream fused GDN-2 kernel.

Those remain later pretraining-system gates. The purpose of this harness is to decide whether the existing model path is numerically and operationally suitable enough to proceed to integrated smoke pretraining.
