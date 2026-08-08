---
status: current
last_reviewed: 2026-08-08
---

# GDN-2 FLA FP32 qualification — current gate

## Current conclusion

The August 8 live Tesla T4 investigation found that the previously reported decay-dependent `fla-core` GDN-2 backward failures were being attributed by an invalid diagnostic oracle.

The old sweep called the adaptive PyTorch reference inside the trainer's CUDA FP16 autocast context. Although the reference explicitly converted recurrence tensors to FP32, eligible matrix multiplications inside the reference could still be autocast to FP16. The failing rows observed on the live notebook showed non-finite **reference** gradients while the corresponding FLA gradients were finite. The old sweep also did not reseed layer initialization per decay, so the reported non-monotonic pattern compared different random layers across rows.

The corrected qualification now:

- disables CUDA autocast only inside the adaptive recurrence oracle;
- keeps the surrounding Small-LLM layer under the real trainer contract: FP32 parameters + CUDA FP16 autocast;
- uses deterministic layer seed `20260808` and input/upstream seed `12345` for every row;
- counts a failure against FLA only when the FP32 adaptive reference is itself finite;
- persists the JSON report.

Single synthetic entry point remains:

```text
python kaggle/run_gdn2_fla_fp32.py
```

Qualified runtime:

```text
GPU: Tesla T4 / SM75
PyTorch: 2.10.0+cu128
CUDA runtime: 12.8
Triton: 3.6.0
fla-core: 0.5.2
saved gdn_chunk_size: 32
FLA runtime chunk: 64
```

## Corrected synthetic result

Decay sweep:

```text
-0.25
-0.5
-0.75
-1.0
-1.25
-1.5
-2.0
-3.0
-4.0
-5.0
-6.0
```

Result:

```text
mixed FLA passing: all 11 points
mixed FLA failing: []

full-FP32 FLA passing: all 11 points
full-FP32 FLA failing: []

invalid FP32 reference rows: []
```

Because mixed FLA no longer reproduces a candidate-specific failure against the corrected oracle, the evidence does **not** support the hypothesis that full-FP32 execution fixed a mixed-precision FLA kernel instability. Full-FP32 passes, but mixed FLA also passes.

Raw evidence:

```text
llm_docs/evidence/gdn2_fla_fp32_qualification_corrected_2026-08-08.json
```

## Real step-4000 gate

The verified remote checkpoint was restored and matched to the attached 500M dataset:

```text
checkpoint: step-00004000
global_step: 4000
last_consumed_block_id: 3999
next block: 4000
next production update: 4001
microbatch: 4
block: 16 x 2048
checkpoint GradScaler scale: 256.0
```

`kaggle/run_gdn2_fla_step4000_parity.py` reproduced the real trainer forward/backward contract over the **entire next block**, including checkpoint loss scaling, while deliberately stopping before clipping or optimizer/scheduler/data mutation.

Both candidates passed against the finite FP32 adaptive reference:

```text
mixed FLA:
  forward parity: PASS
  all gradients finite: PASS
  all parameter gradient parity: PASS
  gradient failures: 0

full-FP32 FLA:
  forward parity: PASS
  all gradients finite: PASS
  all parameter gradient parity: PASS
  gradient failures: 0

REAL_STEP_4000_PARITY: PASS
optimizer step executed: NO
```

Raw evidence:

```text
llm_docs/evidence/gdn2_fla_step4000_parity_2026-08-08.json
```

## Warmed throughput

On the true block 4000 after JIT/autotune warmup:

```text
adaptive FP32 recurrence: 1964.75 target tok/s
FLA mixed:               22765.80 target tok/s   (11.587x adaptive)
FLA full FP32:           21244.76 target tok/s   (10.813x adaptive)
```

All measured backward passes remained finite.

Raw evidence:

```text
llm_docs/evidence/gdn2_fla_step4000_benchmark_2026-08-08.json
```

## Production implication

The active exact-semantics backend selected by the evidence is **mixed FLA on `fla-core==0.5.2`**, not full-FP32 FLA:

- it passes the corrected synthetic sweep;
- it passes the real checkpoint/full-next-block forward and all-gradient parity gate;
- it is the fastest tested backend;
- it preserves the recurrence equation, learned decay, checkpoint keys, and saved `gdn_chunk_size=32`.

The production dependency declaration is aligned to `fla-core==0.5.2` only after this qualification. No diagnostic executed or accepted update 4001.

The detailed evidence and historical correction are in:

```text
llm_docs/evidence/gdn2_fla_corrected_oracle_and_step4000_qualification_2026-08-08.md
```
