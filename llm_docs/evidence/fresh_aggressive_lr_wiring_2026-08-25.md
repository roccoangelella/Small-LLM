# Fresh aggressive LR wiring audit — 2026-08-25

## 100M/2B 10% SFT

The accepted `100m-2b-sft-s0-10pct-001` training path is now isolated from the historical 4% SFT implementation and uses a fresh-from-token-zero aggressive WSqD policy.

Frozen LR landmarks:

```text
peak                         3.0e-5
post-settle                  1.0e-5
terminal-cooldown start      1.0e-6
final                        5.0e-7
```

Frozen phase geometry:

```text
0-5%       linear warmup
5-8%       cosine settle
8-96%      calibrated power-law decay
96-100%    linear terminal cooldown
```

The published training split contains 200,099,738 packed loss-bearing targets. The shared planner resolves this exact horizon to:

```text
warmup targets               10,004,986
settle targets                6,002,992
settle end                   16,007,978
cooldown start              192,095,749
cooldown targets              8,003,989
final horizon               200,099,738
calibrated power exponent    ~0.9266283828
```

`trainer/fresh_decay.py` owns the reusable phase/ratio planner. `trainer/config.py` now supports fresh WSqD by requiring the WSqD LR anchor to equal the warmup endpoint, while historical continuation mode remains warmup-free. `trainer/schedule.py` applies the warmup before the existing settle/power/cooldown mechanics.

`post_training/sft/config.py` retains `build_s0_trainer_config()` as the historical WSD constructor and adds `build_s0_aggressive_trainer_config()` for the accepted 10% run. The aggressive constructor fails closed if peak LR is not exactly `3e-5` and verifies the three lower LR landmarks.

The dual-T4 10% run executes through `kaggle/dual_t4_sft_10pct.py`, a narrow wrapper around the already-qualified `dual_t4_sft.py` execution path. The controlling runtime pins only this 10% training path to implementation commit:

```text
9e0d231f9cf4c16dea94e300ca62377444559355
```

The completed 4% SFT path retains its historical implementation pin and WSD schedule.

## Fresh pretraining audit

The source audit found **no already-registered next fresh pretraining trajectory** in the current roadmap. The active 100M/10B trajectory is `100m-10b-deep-decay-from-step15500`, which is an explicit continuation and must retain its checkpointed continuation schedule.

The generic historical finite/incremental dataset planning helpers still contain WSD contracts for the already-built scaling datasets. Those contracts are historical/reproducibility boundaries and were not silently rewritten.

ADR 0126 now makes the shared `trainer/fresh_decay.py` planner mandatory when the next fresh pretraining trajectory is defined. At pretraining LR scale its frozen landmarks are:

```text
peak                         3.0e-4
post-settle                  1.0e-4
terminal-cooldown start      1.0e-5
final                        5.0e-6
```

The same 5% / 3% / long calibrated power-law / 4% geometry applies from token zero, with the exponent recalibrated against the exact future target horizon.

Therefore the audit result is deliberately not phrased as “the next run already has it”: there is no next fresh run profile to verify yet. What is now frozen is the implementation primitive and decision gate that the next profile must consume.

## Verification limitations

The project execution connector was unavailable in this session, so no VPS/Kaggle runtime test was executed. Focused regression tests were added for fresh WSqD, SFT LR landmarks, and the 10% runtime pin/wrapper. GitHub Actions triggered after the commits but its `unit-tests` job again failed before allocating a runner or exposing any steps (`steps=[]`), so it provides no positive or negative code-test signal.
