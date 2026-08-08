---
status: current
last_reviewed: 2026-08-08
---

# Current GDN-2 backend qualification status

## Problem and diagnosis

The completed approximately-20M / 100M run slowed from roughly 3,830 target tok/s early to roughly 445 target tok/s late, with validation slowing by almost the same factor. Data loading was not the bottleneck. The leading explanation is that stronger learned GDN-2 decay makes the correctness-first adaptive PyTorch backend repeatedly subdivide chunks and synchronize with Python, destroying GPU efficiency while preserving the intended recurrence.

Standalone FLA qualification on Tesla T4 strongly supports that diagnosis rather than a need to clip learned decay.

## Standalone T4 qualification

Environment:

```text
Tesla T4, compute capability 7.5
PyTorch 2.10.0+cu128
CUDA 12.8
Triton 3.6.0
fla-core 0.5.1
flash-linear-attention 0.5.1
```

Qualification summary:

```text
forward_correctness: True
backward_correctness: True (normal-decay FP16 gradient parity)
FLA speedup over adaptive, normal forward: 20.830x
FLA speedup over adaptive, strong-decay forward: 162.541x
adaptive strong-decay forward retention: 0.086x
FLA strong-decay forward retention: 0.671x
FLA speedup over adaptive, strong-decay forward+backward: 135.441x
```

Raw backend rates at batch 4 / context 2,048:

```text
normal:
  adaptive forward:             177,861 tok/s
  FLA forward:                3,704,843 tok/s
  adaptive forward+backward:    58,623 tok/s
  FLA forward+backward:       1,062,705 tok/s

strong log_decay=-6:
  adaptive forward:              15,296 tok/s
  FLA forward:                2,486,218 tok/s
  adaptive forward+backward:      6,050 tok/s
  FLA forward+backward:         819,392 tok/s
```

Interpretation:

- Forward recurrence parity passes for normal decay, `log_decay=-6`, and extreme `log_decay=-10`.
- Normal-decay backward gradients for q, k, v, log-decay, erase, write, and initial state match the recurrent oracle within the probe tolerance.
- The full FLA strong-decay forward+backward path executes successfully and is about 135.4x faster than the adaptive backend in the same stress case.
- The adaptive backend keeps only about 8.6% of normal forward speed in the strong-decay stress regime; FLA keeps about 67.1%.
- FLA is already about 20.8x faster in the normal forward case, so the PyTorch reference backend is intrinsically expensive even before pathological splitting begins.
- Strong learned decay should not be clipped merely to protect the old backend.

The standalone probe only performs recurrent-oracle gradient parity for the normal-decay backward case. Strong-decay gradient parity is therefore part of the integration-level gate.

## Integration now implemented on `main`

ADR 0018 authorizes a checkpoint-compatible integration experiment. The implementation keeps the complete Small-LLM GDN-2 layer and swaps only recurrence execution on the supported CUDA path.

New integration structure:

```text
StableGatedDeltaNet2
  same q/k/v projections
  same convolutions
  same learned decay
  same erase/write gates
  same output gate/norm/projection
  same state-dict keys
        |
        +-- CUDA + chunk_size=64 -> fla.ops.gdn2.chunk_gdn2
        |
        +-- CPU / unsupported test geometry -> AdaptiveChunkwiseGDN2Backend
```

Files:

- `model/gdn2_fla.py` — lazy FLA adapter and call-time preferred backend;
- `model/gdn2_stable.py` — assembled GDN layer now uses the preferred wrapper by default;
- `tests/test_gdn2_stable.py` — CPU fallback, assembly, and checkpoint-key invariance tests;
- `pyproject.toml` — optional `fla` extra pins `fla-core==0.5.1`;
- `kaggle/run_gdn2_fla_layer_probe.py` — one-click whole-layer and optional checkpoint probe.

On the active 64-token CUDA path FLA is mandatory. Missing FLA or a kernel failure is raised rather than silently reverting to the pathological adaptive training path. This prevents an apparently healthy run from accidentally returning to the slow backend.

## Next integration gate

Run on the T4:

```bash
git pull --ff-only
python kaggle/run_gdn2_fla_layer_probe.py
```

The probe compares the entire Small-LLM GDN layer, not just the bare recurrence. It checks output and all parameter/input gradients under both normal decay and a forced approximately `log_decay=-6` regime.

To check an existing trainer checkpoint as well:

```bash
python kaggle/run_gdn2_fla_layer_probe.py --checkpoint /path/to/checkpoint.pt
```

Checkpoint mode strict-loads the same trainer checkpoint into an adaptive-reference full model and the integrated FLA full model, then compares short full-model logits.

## Decision boundary

- Do **not** clip or bound learned GDN-2 decay based on the slowdown evidence.
- The FLA integration is implemented for qualification and checkpoint inspection.
- Do **not** authorize a fresh production 500M restart yet; the user will inspect integrated/checkpoint behavior first and decide separately.
- Existing checkpoints are expected to remain load-compatible because the FLA backend introduces no learned parameters or state-dict entries.
- Switching backend is mathematically compatible but not expected to preserve bitwise-identical future training trajectories because floating-point operation ordering differs.

Detailed standalone evidence: [`../evidence/gdn2_fla_t4_full_probe_2026-08-08.md`](../evidence/gdn2_fla_t4_full_probe_2026-08-08.md)

Qualification decision: [`../decisions/0016-qualify-fla-gdn2-before-changing-decay.md`](../decisions/0016-qualify-fla-gdn2-before-changing-decay.md)

Integration decision: [`../decisions/0018-integrate-fla-gdn2-as-checkpoint-compatible-cuda-backend.md`](../decisions/0018-integrate-fla-gdn2-as-checkpoint-compatible-cuda-backend.md)
