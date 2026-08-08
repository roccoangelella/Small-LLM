# GDN-2 FLA corrected-oracle and real step-4000 qualification — 2026-08-08

## Scope

This evidence closes the August 8 FLA GDN-2 reliability investigation for the active 20M/500M trajectory without changing recurrence semantics, learned decay, checkpoint keys, or the saved `gdn_chunk_size=32` configuration.

The live runtime was the user-authorized Kaggle Jupyter notebook on Tesla T4 / SM75. The credential-bearing notebook URL is intentionally not recorded here.

Environment:

```text
GPU: Tesla T4 / SM75 (two T4s visible; one GPU used per scientific process)
PyTorch: 2.10.0+cu128
CUDA runtime: 12.8
Triton: 3.6.0
fla-core: 0.5.2
outer trainer contract: FP32 master parameters + CUDA FP16 autocast
saved/configured GDN chunk: 32
FLA runtime chunk: 64
```

## 1. The previous decay-sweep oracle was invalid under CUDA autocast

The first execution of the previously prepared FP32 qualification script reproduced many historical `FAIL` rows. Inspection of the failure side showed that the non-finite gradients were in the **adaptive reference**, while FLA's corresponding gradients were finite.

Representative original rows included:

```text
g=-0.50:
  x              ref_nonfinite=8192   fla_nonfinite=0
  A_log          ref_nonfinite=1      fla_nonfinite=0
  dt_bias        ref_nonfinite=6      fla_nonfinite=0
  q_proj.weight  ref_nonfinite=16384  fla_nonfinite=0

g=-0.75:
  x              ref_nonfinite=8960   fla_nonfinite=0
  q_proj.weight  ref_nonfinite=65536  fla_nonfinite=0
  k_proj.weight  ref_nonfinite=65536  fla_nonfinite=0
```

The same side inversion appeared in the original full-FP32-candidate phase: the FLA gradients remained finite while the reference became non-finite at several decay points.

Root cause in the diagnostic harness:

1. the adaptive reference explicitly converts recurrence tensors to FP32, but it was called inside the outer CUDA FP16 autocast context;
2. eligible matrix multiplications inside that reference could therefore be autocast back to FP16 despite the `.float()` conversions;
3. the previous sweep seeded source/upstream tensors but did **not** reseed layer initialization for each row, so each decay point also tested a different random layer initialization and the reported non-monotonic pass/fail pattern was not process-reproducible.

This means the historical v0.5.1/v0.5.2 synthetic decay-sweep failures that did not distinguish reference-side from FLA-side non-finiteness are not valid evidence of a released FLA backward failure. They remain preserved as historical evidence and are not deleted or rewritten.

## 2. Diagnostic correction

`kaggle/run_gdn2_fla_fp32_qualification.py` was changed only as a diagnostic/oracle correction:

- the adaptive PyTorch recurrence executes with CUDA autocast disabled internally;
- the surrounding Small-LLM layer still uses the trainer's FP16 autocast contract;
- layer initialization is reset to deterministic seed `20260808` for every decay and both candidate modes;
- source/upstream tensors use deterministic seed `12345`;
- a row can count as a candidate failure only when the FP32 adaptive reference itself is valid;
- reference-invalid rows invalidate the experiment rather than being attributed to FLA;
- the report is persisted as JSON.

A separate reference-only T4 check confirmed finite output and finite gradients at every tested constant decay from `-0.25` through `-6.0`.

## 3. Corrected synthetic decay sweep

Command:

```text
python kaggle/run_gdn2_fla_fp32.py
```

Corrected `fla-core==0.5.2` result:

```text
mixed FLA passing decay:
[-0.25, -0.5, -0.75, -1.0, -1.25, -1.5, -2.0, -3.0, -4.0, -5.0, -6.0]

mixed FLA failing decay:
[]

full-FP32 FLA passing decay:
[-0.25, -0.5, -0.75, -1.0, -1.25, -1.5, -2.0, -3.0, -4.0, -5.0, -6.0]

full-FP32 FLA failing decay:
[]

invalid reference rows:
[]
```

Worst observed finite gradient absolute difference in the corrected synthetic sweep was approximately:

```text
mixed FLA:     2.344e-02
full-FP32 FLA: 3.906e-03
```

The mixed baseline therefore does **not** reproduce a candidate-specific kernel failure once compared against a finite FP32 oracle. Full-FP32 also passes every synthetic point, but the evidence does not support claiming that full-FP32 "fixed" a mixed-precision kernel bug; the earlier failure attribution was a harness error.

Raw report:

```text
llm_docs/evidence/gdn2_fla_fp32_qualification_corrected_2026-08-08.json
sha256: 3996362c162228836bd8fb7e02b916ff5629dda92e9219eff519d82886189935
```

## 4. Verified real checkpoint and exact next block

The existing remote restore path was used to restore the latest verified 500M checkpoint and cryptographically match it against the attached private dataset manifest.

Verified state:

```text
checkpoint_id: step-00004000
global_step: 4000
last_consumed_block_id: 3999
next block: 4000
next optimizer update if production resumes: 4001
microbatch_size: 4
block geometry: 16 x 2048
target tokens in block: 32768
checkpoint GradScaler scale: 256.0
```

No optimizer step was executed by the qualification.

## 5. Real step-4000 / block-4000 parity gate

`kaggle/run_gdn2_fla_step4000_parity.py` performs one complete accumulated training backward over the true next block with:

- the checkpoint model weights;
- the checkpoint's real next data block;
- FP32 master parameters;
- CUDA FP16 autocast;
- microbatch 4;
- the checkpoint GradScaler scale `256.0` applied before backward and unscaled before gradient comparison;
- the same summed cross-entropy / full-block target normalization as the trainer;
- no clipping, optimizer step, scheduler step, data acknowledgement, W&B write, or checkpoint mutation.

It compares both current mixed FLA and full-FP32 FLA against the finite FP32 adaptive recurrence oracle.

Result:

```text
REAL_STEP_4000_PARITY: PASS

mixed FLA:
  forward parity: PASS
  all gradients finite: PASS
  all parameter gradient parity: PASS
  loss: 3.907714068889618
  |loss-reference|: 5.161762237548828e-05
  maximum full-logit absolute difference: 0.078125
  maximum parameter-gradient absolute difference: 0.000125885009765625
  gradient failures: 0

full-FP32 FLA:
  forward parity: PASS
  all gradients finite: PASS
  all parameter gradient parity: PASS
  loss: 3.9077218174934387
  |loss-reference|: 4.38690185546875e-05
  maximum full-logit absolute difference: 0.0625
  maximum parameter-gradient absolute difference: 0.000133514404296875
  gradient failures: 0

FP32 adaptive reference loss: 3.9077656865119934
```

Raw report:

```text
llm_docs/evidence/gdn2_fla_step4000_parity_2026-08-08.json
sha256: e2b6ed6ee9d50878af11cdc3e147ec6794098a7852dccbc8bcef934d4e7779ca
```

## 6. Warmed real-block throughput

After compilation/autotuning was warm, `kaggle/run_gdn2_fla_step4000_benchmark.py` benchmarked the same real block with forward/backward only and no parity-copy instrumentation.

Two measured repeats per backend:

```text
adaptive FP32 recurrence:
  16.5891 s
  16.7668 s
  median: 1964.75 target tok/s
  peak allocated: 8.94 GB

FLA mixed:
  1.43426 s
  1.44445 s
  median: 22765.80 target tok/s
  peak allocated: 7.89 GB
  speedup vs adaptive: 11.587x

FLA full FP32:
  1.53569 s
  1.54912 s
  median: 21244.76 target tok/s
  peak allocated: 8.12 GB
  speedup vs adaptive: 10.813x
```

All benchmark backward passes produced finite gradients.

Raw report:

```text
llm_docs/evidence/gdn2_fla_step4000_benchmark_2026-08-08.json
sha256: fe2210bdaab0e53ddf2c2d9582e5be8c870be7f5e5f1481da11b4d2c5a9a60a4
```

The current mixed FLA path is both the fastest tested exact-semantics backend and passes the corrected synthetic and real-checkpoint gates. Full-FP32 remains a useful diagnostic/fallback mode but is not required by the observed evidence for the active trajectory.

## 7. Production dependency alignment

The qualified runtime is `fla-core==0.5.2`. Production runtime declarations were therefore updated from `0.5.1` to `0.5.2` only **after** the live T4 qualification succeeded. This is a production alignment with the qualified backend, not a notebook-bootstrap workaround.

Checkpoint/model compatibility is unchanged:

```text
saved gdn_chunk_size: 32
FLA internal runtime chunk: 64
state-dict keys: unchanged
learned decay parameterization: unchanged
recurrence equation: unchanged
no decay clipping/bounding: added none
```

## 8. Conclusion

For the active Tesla T4 / step-4000 trajectory, the observed decay-dependent FLA backward failure was a qualification-harness false positive caused by an autocast-contaminated adaptive reference plus unseeded layer initialization.

With a finite deterministic FP32 oracle, released `fla-core==0.5.2` mixed GDN-2 passes the complete requested synthetic decay sweep and the true step-4000/block-4000 full forward/backward gate, and is approximately `11.59x` faster than the adaptive backend in the warmed real-block benchmark.

This evidence qualifies the existing exact-semantics mixed FLA backend for production continuation from `step-00004000`, provided the production launcher is pinned to the implementation commit that declares `fla-core==0.5.2`. Update 4001 itself has still not been executed or accepted by these diagnostics.
