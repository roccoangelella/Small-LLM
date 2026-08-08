---
status: current
last_reviewed: 2026-08-08
---

# GDN-2 FLA FP32 qualification — current gate

Released FLA v0.5.1/v0.5.2 mixed-precision `chunk_gdn2` training remains blocked for the active 20M/500M trajectory because trainer-AMP backward failures overlap the real verified step-4000 decay regime.

A bounded diagnostic is now authorized by ADR 0020: test whether forcing the complete FLA GDN-2 chunk execution to FP32 removes the synthetic decay-dependent backward failure without changing recurrence or learned-state semantics.

Single Kaggle entry point:

```text
python kaggle/run_gdn2_fla_fp32_qualification.py
```

The script:

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

Safe accepted production state remains:

```text
checkpoint: step-00004000
last_consumed_block_id: 3999
next update: 4001
no FLA update accepted
```

The adapter's new `force_fp32` mode is opt-in. Default assembled-model behavior and production dependency pins are unchanged.
