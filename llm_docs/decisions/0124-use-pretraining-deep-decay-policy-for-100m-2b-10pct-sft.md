---
status: accepted
date: 2026-08-25
---

# ADR 0124: use the pretraining deep-decay policy for the 100M/2B 10% SFT run

## Decision

The accepted 100M/2B 10% S0 SFT run (`100m-2b-sft-s0-10pct-001`) must not use the previous simple SFT WSD schedule with a long flat `3e-5` phase followed by a terminal cosine decay.

Instead, use the same learning-rate policy family adopted for the current 100M/10B deep-decay pretraining continuation:

1. a short cosine settling phase from the SFT peak LR to a lower settle LR;
2. a long monotonic power-law decay;
3. a short terminal linear cooldown to the final LR floor.

This is a schedule-policy decision. The SFT run must use lower LR anchors than pretraining because it starts from an already pretrained model and performs assistant-target SFT rather than base pretraining.

## Re-anchoring requirement

Do not copy the pretraining schedule's absolute token anchors literally. The pretraining deep-decay schedule is a continuation anchored at an already-consumed ~2.03B targets, whereas this SFT optimizer starts its own target-token clock at zero and has only ~200.10M packed targets.

The SFT implementation must therefore preserve the settle -> power-law -> cooldown policy while re-solving its token anchors and, if necessary, the power exponent against the actual SFT horizon. The intended LR anchor ratios should be preserved unless a later decision explicitly changes them.

## LR anchor selection status

The exact SFT LR anchors are not frozen by this ADR. They are being selected separately before launch.

The leading candidate is ratio-preserving 0.1x scaling of the pretraining anchors, because the existing SFT peak (`3e-5`) is already exactly one tenth of the pretraining peak (`3e-4`):

```text
                       pretraining       candidate SFT
peak                    3e-4              3e-5
post-settle             1e-4              1e-5
cooldown start          1e-5              1e-6
final                    5e-6              5e-7
```

These candidate values are evidence/recommendation, not yet a separately accepted numeric freeze. Do not launch the 10% SFT run until the LR anchors are explicitly selected and wired.

## Rationale

The 10% SFT corpus is roughly 2.5x larger than the completed 4% S0 corpus. Holding LR flat near the peak for most of that larger horizon is no longer the desired policy. A long continuous decay should progressively reduce parameter movement as instruction behavior is acquired, while the final cooldown provides a low-LR landing phase.

Keeping the policy family shared with pretraining also makes scheduler behavior easier to reason about and audit across the project, while independently scaling the LR magnitude for the post-training regime.
