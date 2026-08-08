---
status: current
last_reviewed: 2026-08-08
---

# GDN-2 FLA FP32 qualification — current gate

Released FLA v0.5.1/v0.5.2 mixed-precision `chunk_gdn2` training remains blocked for the active 20M/500M trajectory because trainer-AMP backward failures overlap the real verified step-4000 decay regime.

A bounded diagnostic is authorized by ADR 0020: test whether forcing the complete FLA GDN-2 chunk execution to FP32 removes the synthetic decay-dependent backward failure without changing recurrence or learned-state semantics.

Single notebook entry point:

```text
python kaggle/run_gdn2_fla_fp32.py
```

The wrapper self-provisions the diagnostic runtime and then launches:

```text
kaggle/run_gdn2_fla_fp32_qualification.py
```

The scientific script:

1. forces diagnostic `fla-core==0.5.2`;
2. runs the known mixed-precision full-layer decay sweep as an in-run control;
3. repeats the identical sweep with the FLA GDN-2 execution forced to FP32 internally;
4. checks output finiteness/parity and every tested input/parameter gradient against the adaptive PyTorch reference;
5. prints a bounded JSON report between `COPY_PASTE_REPORT_BEGIN` and `COPY_PASTE_REPORT_END`.

The experiment is synthetic and does not load the checkpoint, start the trainer, or perform optimizer updates.

Production remains blocked regardless of the synthetic result. A successful result requires:

```text
mixed-precision baseline: reproduces >=1 known failure
full-FP32 candidate:       passes every tested decay point
```

If that gate passes, the next required experiment is direct forward/backward gradient parity on the verified real step-4000 checkpoint and next real training microbatch. Only after that may a disposable optimizer-update test be considered.

## Notebook bootstrap history

Two bootstrap-only failures occurred before any scientific FLA result was produced:

1. the first entry point installed `fla-core==0.5.2 --no-deps` into a runtime without Triton, so import failed with `ModuleNotFoundError: No module named 'triton'`;
2. the first self-provisioning wrapper then discovered that the notebook had `torch=2.11.0+cpu`, so there was no CUDA PyTorch/Triton stack to run FLA at all.

Neither failure loaded the checkpoint or exercised FLA forward/backward correctness.

The wrapper is now pinned to repair this specific environment mismatch. When an NVIDIA GPU is visible but PyTorch is CPU-only, it installs the official CUDA 12.8 `torch==2.10.0` wheel, matching the previously qualified Tesla T4 stack (`torch 2.10.0+cu128`, Triton 3.6.0). It verifies CUDA in a fresh child process before launching the scientific diagnostic. If no NVIDIA GPU is visible, it stops and asks for a GPU runtime rather than modifying packages blindly.

## PiLink agent delegation

The user has delegated the active FLA reliability investigation to an agent with PiLink access so that it can work directly against the live notebook/runtime instead of relying on copied logs. The agent should first recover any previously shared Kaggle/SSH endpoint from its authorized memory if available, connect to the live GPU runtime, and execute the existing FP32 qualification gate. If FP32 still fails, it should localize the first non-finite FLA backward intermediate and implement the narrowest exact-recurrence numerical fix, preserving checkpoint/model semantics and production safety boundaries. Any successful change must be committed to `main` with evidence under `llm_docs/`.

Safe accepted production state remains:

```text
checkpoint: step-00004000
last_consumed_block_id: 3999
next update: 4001
no FLA update accepted
```

The adapter's `force_fp32` mode remains opt-in. Default assembled-model behavior and production dependency pins are unchanged.
