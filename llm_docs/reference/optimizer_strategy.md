# Optimizer strategy

_Last reviewed: 2026-08-13_

## Current pretraining optimizer

The project production pretraining optimizer is **hybrid whole-matrix Muon + AdamW**. Pure AdamW remains an explicit control, not the default production path.

Routing is by semantic parameter role and fails closed on an unclassified new trainable parameter.

### Muon branch

Muon receives ordinary two-dimensional feature-transform matrices, including SwiGLU matrices, gated-MHA Q/K/V/gate/output projections, and the ordinary two-dimensional GDN-2 projection matrices. The production recipe orthogonalizes each complete logical matrix; it does not silently fuse unrelated matrices or split Q/K/V by head.

Current recipe identity includes:

```text
Nesterov momentum: 0.95
Newton-Schulz: 8 aggressive + 2 stabilizing iterations
qualification target update RMS: 0.18
Muon weight decay: 0.1
Muon momentum / orthogonalization state: FP32
```

The exact Newton-Schulz coefficients and routed parameter-name list are serialized by the implementation/checkpoint recipe and must match on resume.

### AdamW branch

AdamW receives roles that should not be matrix-orthogonalized, including:

- tied token embedding / LM-head matrix;
- RMSNorm scales;
- biases and reference-required learned offsets;
- GDN-2 `A_log` and `dt_bias`;
- GDN-2 depthwise temporal convolution kernels.

AdamW first/second moments remain FP32. Current weight decay is 0.1 subject to the implementation's explicit no-decay exceptions.

## Shared update contract

Both optimizer branches belong to one atomic optimizer object/update boundary:

1. accumulate the full prepared block;
2. resolve FP16 scaling/non-finite state;
3. unscale once;
4. apply one global gradient clip at 1.0;
5. update Muon and AdamW branches together;
6. commit scheduler/tokens/dataset cursor only after success.

A partial Muon-only or AdamW-only update is illegal.

## Schedule and LR

Muon and AdamW share the run's token-count schedule, with the configured Muon group LR multiplier preserved by the scheduler. Peak LR, exact multiplier, and WSD phase boundaries are **run/profile-specific scientific configuration**, not unresolved global optimizer design questions. Resume restores/rejects drift through checkpointed trainer/optimizer configuration.

ADR 0057/0058 define the current 100M/10B schedule horizon separately from the optimizer-family choice.

## Checkpoint contract

Optimizer state includes FP32 Muon momentum, AdamW moments/steps, parameter groups, LR scales, exact routing identity, and the versioned Muon recipe. A checkpoint created with pure AdamW cannot silently resume as hybrid Muon + AdamW or vice versa.

## Research boundary

The project originally selected this design after reviewing post-2025 Muon recipes and kept per-head Q/K/V Muon and other optimizer families as later controlled ablations. Those research comparisons are rationale, not current authorization to change routing. Any optimizer-family or routing change requires an explicit controlled decision and fresh qualification.
