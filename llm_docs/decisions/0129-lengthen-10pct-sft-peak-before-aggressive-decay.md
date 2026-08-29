---
status: accepted
date: 2026-08-29
---

# ADR 0129: Lengthen the 100M/2B 10% SFT peak before aggressive decay

## Decision

Rerun the frozen 100M/2B 10% S0 corpus as a new scientific trajectory with the same optimizer, same `3e-5` peak learning rate, and the same low terminal learning-rate landmarks, but keep the optimizer at peak LR for materially longer before the aggressive decay begins.

The accepted LR geometry over the exact loss-bearing target-token horizon is:

- warmup: first 5%, `0 -> 3e-5`;
- peak hold: next 15%, constant `3e-5`;
- settle: next 3%, `3e-5 -> 1e-5`;
- aggressive calibrated power-law decay: from 23% through 96%, `1e-5 -> 1e-6`;
- terminal cooldown: final 4%, `1e-6 -> 5e-7`.

For the published train split with `200,099,738` observed loss-bearing targets, the nominal token landmarks are therefore:

```text
warmup end:       10,004,986
peak-hold end:    40,019,946
settle end:       46,022,938
cooldown start:  192,095,749
full horizon:    200,099,738
```

The peak LR remains frozen at `3e-5`. This decision does **not** raise the peak. The experiment changes peak duration only, preserving a long aggressive decay phase so the run still reaches `1e-6` before the final cooldown and `5e-7` at completion.

## Scientific rationale

The completed historical 4% S0 trajectory held `3e-5` for most of training and reduced held-out SFT loss much faster, but moved farther away from the pretrained distribution. The first 10% aggressive-WSqD trajectory instead settled from `3e-5` to `1e-5` immediately after its 5% warmup and then decayed for nearly the entire remaining run. It retained more pretrained capability, but its strict instruction-behavior result did not improve over the 4% run despite substantially more SFT data.

The controlled inference is that the first 10% schedule likely reduced optimization pressure too early. There is not yet evidence that a peak above `3e-5` is needed. A 15% peak hold is selected as an intermediate plasticity regime: materially longer than the first 10% trajectory's effectively zero-length peak plateau, but far shorter than the historical 4% trajectory's roughly 75% high-LR stable phase. The remaining 80% of the run is still devoted to settling plus aggressive decay/cooldown.

This preserves the experimental objective of combining stronger early SFT adaptation with aggressive late-stage convergence rather than conflating peak magnitude and peak duration in the same ablation.

## Artifact identity and isolation

Reuse the already-published immutable 10% dataset:

```text
Kaggle dataset slug: small-llm-100m-2b-sft-s0-10pct-001
train manifest SHA-256: feefc3244bd8a2f369eec85e4a95410c2daf479016c04cf02c8042ca5a4010d3
observed train loss-bearing targets: 200,099,738
```

Do not resume or overwrite the completed first 10% trajectory. The rerun gets a fresh checkpoint/W&B identity:

```text
run ID:   100m-2b-sft-s0-10pct-longpeak-001
W&B ID:   100m-2b-sft-s0-10pct-longpeak-001
```

The dedicated Kaggle 10% runtime is pinned to implementation commit `fd784ed1bb056dc8a2d29a3847e606b8762cecc1`, which contains the fresh-WSqD peak-hold support and the SFT-specific 15% hold. The generic fresh aggressive-decay planner retains a zero peak-hold default, so this decision does not silently alter fresh pretraining schedules.

## Qualification gate

Treat this as a schedule-shape ablation, not an automatic promotion. After completion, compare at minimum:

1. pretrained 100M/2B;
2. historical 4% S0;
3. completed first 10% aggressive S0;
4. new 10% long-peak S0.

Promotion must consider frozen general eval-core retention and instruction behavior together with held-out SFT loss. A lower in-distribution SFT loss alone is insufficient evidence of better model quality.
