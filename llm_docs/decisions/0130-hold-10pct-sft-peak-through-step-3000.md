---
status: accepted
date: 2026-08-29
supersedes: 0129
---

# ADR 0130: Hold the 100M/2B 10% SFT peak through step 3000

## Decision

Supersede ADR-0129 before launching its proposed trajectory. The next 100M/2B 10% S0 rerun will use the same frozen `3e-5` peak LR and the same aggressive low-LR landmarks, but it will warm up substantially faster and hold peak LR through optimizer update 3000.

The accepted step-anchored geometry is:

- updates 1-64: warm up from approximately zero to `3e-5`;
- updates 64-3000: hold `3e-5`;
- updates 3001-3128: fast cosine settle from `3e-5` to `1e-5`;
- after update 3128 through the 96%-of-target-token point: calibrated power-law decay from `1e-5` to `1e-6`;
- final 4% of loss-bearing target tokens: linear cooldown from `1e-6` to `5e-7`.

The schedule is anchored to actual optimizer-block target counts rather than approximating the requested step boundaries with global fractions. Step 3000 therefore remains at peak LR, and step 3001 is the first update below peak.

For intuition only, using the completed 10% run's 6,219-block / 200,099,738-target horizon with near-uniform block counts gives approximately:

```text
step 64:    ~2.06M target tokens, LR = 3e-5
step 3000:  ~96.53M target tokens, LR = 3e-5
step 3128: ~100.65M target tokens, LR = 1e-5
96% point:  192.10M target tokens, LR = 1e-6
end:        200.10M target tokens, LR = 5e-7
```

The calibrated power exponent is correspondingly steep (about 3.56 under the near-uniform-block approximation), so the second half of the run still performs an aggressive optimization squeeze rather than remaining at high LR until the end.

## Rationale

The user explicitly prefers substantially more time at peak LR than ADR-0129's 15% hold and a faster warmup, while retaining an aggressive decay phase. The completed 4% trajectory showed that sustained `3e-5` can remove SFT loss much more rapidly than the first 10% schedule, whereas the first 10% schedule showed better pretrained-distribution retention after decaying too early.

Holding through step 3000 deliberately moves the new ablation much closer to the high-plasticity side of that tradeoff, while reserving roughly the final half of the run for a fast settle, steep calibrated power-law decay, and terminal cooldown. Peak magnitude remains unchanged so the experiment still isolates schedule shape rather than conflating peak LR and peak duration.

## Artifact isolation

Reuse the immutable published 10% dataset:

```text
dataset slug: small-llm-100m-2b-sft-s0-10pct-001
train manifest SHA-256: feefc3244bd8a2f369eec85e4a95410c2daf479016c04cf02c8042ca5a4010d3
observed loss-bearing train targets: 200,099,738
```

Use a new training/W&B identity so neither the completed first 10% trajectory nor any unlaunched ADR-0129 namespace can be resumed accidentally:

```text
run ID: 100m-2b-sft-s0-10pct-peak3000-001
W&B ID: 100m-2b-sft-s0-10pct-peak3000-001
```

The dedicated Kaggle runtime pins training to implementation commit `caa7fa54fe16510d30ef92eca19d95f86585e20e`, which contains the step-anchored schedule, dedicated 10% wrapper, and its regression tests. Generic fresh pretraining schedules are unchanged.

## Qualification gate

Treat this as another schedule-shape ablation. After completion, compare at minimum pretrained 100M/2B, historical 4% S0, the completed first 10% aggressive S0, and this peak-through-3000 trajectory. Promotion must consider general eval-core retention, strict instruction behavior, qualitative outputs, EOS/runaway/repetition behavior, and held-out SFT loss together; a lower SFT loss alone is not sufficient.
