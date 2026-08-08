---
status: accepted
date: 2026-08-08
---

# ADR 0020 — Qualify FLA GDN-2 with full-FP32 kernel execution

## Context

Released FLA `chunk_gdn2` v0.5.1 and v0.5.2 are dramatically faster than the adaptive PyTorch GDN-2 backend and are forward-correct on the Tesla T4, but trainer-realistic FP32-master + FP16-autocast tests show decay-dependent backward failures. The verified step-4000 checkpoint overlaps tested failing decay regimes, so released mixed-precision FLA training is blocked.

The Small-LLM adapter currently canonicalizes ordinary FLA compute tensors (`q`, `k`, `v`, erase, write) to the low-precision value dtype while keeping log-decay and recurrent state FP32. A plausible remaining cause is mixed-precision numerical instability in the FLA WY/chunk backward path rather than the GDN-2 recurrence itself.

## Decision

Before abandoning FLA or changing learned decay semantics, run one bounded qualification experiment in which the complete FLA GDN-2 chunk execution is forced to FP32 while the surrounding trainer contract remains FP32 parameters + CUDA FP16 autocast.

Implementation requirements:

- add an opt-in `force_fp32` execution mode to the FLA GDN-2 adapter;
- leave the default adapter behavior unchanged and therefore leave production behavior unauthorized;
- preserve checkpoint/model configuration `gdn_chunk_size=32` while FLA continues to execute its internal 64-token chunks;
- do not alter learned parameters, checkpoint keys, recurrence equations, decay parameterization, optimizer state, or scheduler state;
- use released `fla-core==0.5.2` for the diagnostic;
- provide one Kaggle entry point, `kaggle/run_gdn2_fla_fp32_qualification.py`;
- in the same run, first reproduce the current mixed-precision baseline and then repeat the identical full-layer decay sweep with FLA forced to FP32;
- require finite outputs, finite gradients, and gradient parity against the adaptive PyTorch reference at every tested decay point;
- print a bounded copy-paste report for review.

## Qualification boundary

A synthetic FP32 pass is not authorization to resume training. If the FP32 candidate passes every tested decay point and the mixed-precision baseline reproduces at least one known failure, the next required gate is direct forward/backward parity on the real verified step-4000 checkpoint and its next real training microbatch before any optimizer update is accepted.

If the FP32 candidate still fails, do not resume training and instead localize the first non-finite FLA backward intermediate before considering a kernel patch.

## Consequences

- The verified production state remains `step-00004000`, last consumed block `3999`, next update `4001`.
- No FLA optimizer update is authorized by this ADR.
- No decay clipping/bounding is authorized.
- The production dependency pins remain unchanged while this is a diagnostic-only path.
- The experiment can distinguish a mixed-precision failure from a deeper algorithmic/kernel failure without modifying model semantics.
