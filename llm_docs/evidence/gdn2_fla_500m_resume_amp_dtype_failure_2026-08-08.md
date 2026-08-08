# 500M FLA resume attempt — AMP dtype integration failure

Date: 2026-08-08

## Context

After standalone FLA GDN-2 qualification and an initial full-layer parity probe, the existing 20M / 500M run was authorized to resume from its latest verified checkpoint with the checkpoint-compatible FLA execution backend.

The launcher restored the verified checkpoint at global step 4000 and attempted global steps 4001–15264.

## Observed failure

The first resumed training update did not complete. Triton compilation failed inside FLA GDN-2 WY recomputation with:

```text
AssertionError: Both operands must be same dtype. Got fp32 and fp16
triton.compiler.errors.CompilationError
...
b_u = tl.dot(b_A, b_vb)
Both operands must be same dtype. Got fp32 and fp16
```

The run failed closed before a successful update 4001, so the latest verified checkpoint remains step 4000 and no new optimizer/data/scheduler state was committed.

Environment/model evidence from the failed launch:

```text
resume start: global steps 4001-15264
saved gdn_chunk_size: 32
d_model: 256
n_layers: 8
gdn heads: 4
gdn key/value dim: 64
context: 2048
precision: FP16 trainer autocast
```

The unrelated warning `Failed to initialize NumPy: No module named 'numpy'` is not the cause of the GDN failure.

## Root cause

The first integration probe used `model.cuda().half()`, so q/k/v/erase/write all entered FLA as FP16. That did not reproduce the actual trainer precision contract, where master parameters remain FP32 and the forward runs under CUDA FP16 autocast.

In the real trainer path, normalized q/k can be FP32 while v/write remain FP16. FLA v0.5.1 allocates its solved WY matrix `Akk` with `k.dtype`; its recompute kernel later evaluates:

```text
b_A = load(A)                  # dtype follows k
b_vb = (v * write_gate)        # dtype follows v
b_u = tl.dot(b_A, b_vb)
```

Triton requires both dot operands to have the same dtype. With k/A in FP32 and v/write in FP16, compilation fails before training executes.

This is an adapter precision-contract bug, not a recurrence mismatch, checkpoint mismatch, strong-decay failure, or T4 incompatibility.

## Fix on main

`model/gdn2_fla.py` now canonicalizes q, k, v, erase, and write to the value tensor dtype before entering FLA. In the active trainer this yields the already-qualified FP16 compute contract. Log-decay and recurrent state remain FP32. FLA output is cast back to the original Small-LLM q dtype before returning, and the internal casts remain differentiable.

The integration probe was also corrected: it now keeps model parameters in FP32 and runs layer/checkpoint comparisons inside CUDA FP16 autocast rather than converting the whole model to FP16.

The 500M launcher is repinned to the implementation containing the AMP-safe adapter. Historical checkpoints still retain `gdn_chunk_size=32`; CUDA FLA execution still uses its fixed internal chunk size 64.

## Required gate before another resume attempt

Run:

```bash
python kaggle/run_gdn2_fla_layer_probe.py
```

and require:

```text
layer_forward_backward_parity: True
trainer_amp_contract_tested: True
```

Only after that AMP-realistic integration gate passes should `python kaggle/run_20m_500m.py` be retried.
