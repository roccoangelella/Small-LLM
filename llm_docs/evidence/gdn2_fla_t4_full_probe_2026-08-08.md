---
status: evidence
date: 2026-08-08
---

# FLA GDN-2 T4 full probe result

## Environment

```text
GPU: Tesla T4
PyTorch: 2.10.0+cu128
Triton: 3.6.0
fla-core: 0.5.1
flash-linear-attention: 0.5.1
```

## Correctness

Forward parity against the project tokenwise recurrent oracle passed in FP16 for:

- normal decay;
- strong `log_decay=-6`;
- extreme `log_decay=-10`.

Maximum absolute forward differences were small:

```text
normal output: 4.58e-05
normal final state: 2.02e-04
stress -6 output: 3.05e-05
stress -6 final state: 1.19e-04
extreme -10 output: 3.05e-05
extreme -10 final state: 8.28e-05
```

Backward gradient parity against the recurrent oracle passed for the normal-decay FP16 case. Maximum absolute gradient differences:

```text
erase: 1.83e-04
g:     7.14e-04
h0:    4.41e-05
k:     3.91e-03
q:     2.44e-04
v:     1.95e-03
write: 1.95e-03
```

The probe did not perform recurrent-oracle gradient parity for the `log_decay=-6` stress case. It did execute and benchmark the full forward+backward stress path successfully.

## Backend microbenchmarks

The benchmark geometry is the active 20M GDN geometry at batch 4 and context 2048. `backend_tokens_per_second` is for one GDN backend call and is not whole-model training throughput.

### Normal decay

```text
adaptive forward:          177,861 tok/s
FLA forward:             3,704,843 tok/s
adaptive forward+backward: 58,623 tok/s
FLA forward+backward:    1,062,705 tok/s
```

### Strong decay (`log_decay=-6`)

```text
adaptive forward:           15,296 tok/s
FLA forward:             2,486,218 tok/s
adaptive forward+backward:  6,050 tok/s
FLA forward+backward:      819,392 tok/s
```

### Derived ratios

```text
forward FLA speedup over adaptive, normal: 20.830x
forward FLA speedup over adaptive, stress: 162.541x
adaptive forward stress retention: 0.086x
FLA forward stress retention: 0.671x
FLA forward+backward speedup over adaptive stress: 135.441x
```

## Interpretation

This result strongly supports the runtime hypothesis. The current adaptive PyTorch backend collapses under strong decay, retaining only about 8.6% of its normal forward speed. FLA retains about 67.1% of its normal forward speed while preserving forward recurrence parity.

The normal-decay forward+backward result is also about 18.1x faster for FLA than the adaptive backend, and the strong-decay forward+backward path is about 135.4x faster. This shows that the pathological late-training slowdown is primarily an execution-backend problem rather than evidence that learned GDN-2 decay must be clipped.

## Decision boundary after this result

This evidence is sufficient to move from standalone kernel qualification to a checkpoint-compatible Small-LLM integration experiment. It does not by itself authorize changing the active 500M production run.

The next implementation gate should be:

1. add an FLA-backed `GDN2Backend` adapter while preserving existing layer parameters/state-dict keys;
2. run full-layer forward/backward parity against the current layer on representative normal and strong-decay inputs;
3. verify checkpoint load compatibility;
4. run a short optimizer-step replay/mini-training comparison;
5. if those pass, decide whether to restart the 500M experiment from update 1 with the FLA backend or perform an explicitly recorded backend migration.
