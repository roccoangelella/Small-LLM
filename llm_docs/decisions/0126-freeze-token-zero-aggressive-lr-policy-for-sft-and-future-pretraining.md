---
status: accepted
date: 2026-08-25
---

# ADR 0126: freeze the token-zero aggressive LR policy for 10% SFT and future fresh pretraining

## Decision

ADR 0124 selected the settle -> power-law -> terminal-cooldown policy family for the accepted 100M/2B 10% S0 run. ADR 0125 clarified that the step-15,500 continuation anchor is historical evidence, not part of the schedule definition for a fresh optimizer.

This ADR freezes the numeric fresh-run policy.

Every fresh run governed by this policy starts its optimizer LR clock at **token zero** and uses:

1. a 5% linear warmup from the initial low LR to the peak LR;
2. an immediate 3% cosine settle from peak LR to one third of peak;
3. a long calibrated power-law decay with no stable/flat plateau;
4. a terminal linear cooldown over the final 4% of the target-token horizon.

The power exponent is recomputed from each run's exact packed target-token horizon so the power-law phase lands exactly at one thirtieth of peak LR when terminal cooldown begins. Do not copy the continuation experiment's `~1.6270515945` exponent into a fresh run unless the fresh horizon independently produces that value.

The phase geometry is therefore:

```text
0% ---- 5% -------- 8% -------------------------------- 96% ---- 100%
     warmup    settle             power-law                 cooldown
```

There is no continuation-style plateau through ~20% of training.

## 100M/2B 10% SFT LR anchors

The accepted SFT run `100m-2b-sft-s0-10pct-001` uses exactly:

```text
peak LR:                    3.0e-5
post-settle LR:             1.0e-5
LR at cooldown start:       1.0e-6
final LR:                   5.0e-7
```

The published bundle has 200,099,738 packed train targets. Under the frozen percentage geometry this resolves to:

```text
warmup:                     10,004,986 targets
warmup end / WSqD anchor:   10,004,986
settle:                      6,002,992 targets
settle end:                 16,007,978
power-law end / cooldown:  192,095,749
terminal cooldown:           8,003,989 targets
final target horizon:      200,099,738
fresh calibrated p:         ~0.9266283828
```

The SFT implementation must fail closed if the peak LR differs from `3e-5` for this accepted run. The historical 4% SFT configuration remains WSD and is not rewritten.

## Future fresh pretraining LR anchors

The same token-zero policy is the project default for the **next newly defined fresh pretraining trajectory**, scaled to the established pretraining LR magnitude:

```text
peak LR:                    3.0e-4
post-settle LR:             1.0e-4
LR at cooldown start:       1.0e-5
final LR:                   5.0e-6
```

The 5% / 3% / long power-law / 4% geometry is preserved, and the power exponent must be recalibrated from that run's exact target-token horizon.

This decision does **not** mutate any completed or in-flight pretraining run. In particular, `100m-10b-deep-decay-from-step15500` remains the exact step-15,500 continuation defined by ADR 0095/0114 and resumes from its own checkpointed continuation scheduler state.

## Audit of current pretraining launch state

At the time of this decision, no concrete next fresh pretraining run/profile is registered in the current roadmap. The active 100M/10B run is the continuation above. The older generic finite and incremental dataset trainer-plan helpers still describe historical WSD contracts, so it would be incorrect to claim that an already-defined next fresh run was automatically using this policy.

The shared implementation now lives in `trainer/fresh_decay.py` and is tested at both SFT and pretraining LR scales. When the next fresh pretraining profile is created, its launcher/configuration must use this shared token-zero planner rather than the historical WSD plan. Existing dataset/profile contracts remain explicitly historical evidence and must not be silently rewritten.

## SFT implementation boundary

The accepted 10% SFT run uses a dedicated pinned training worktree and wrapper. This isolates the scientific scheduler change from the completed 4% SFT run while preserving the already-qualified dual-T4 execution slicing, hybrid Muon+AdamW optimizer, immutable dataset identity, exact-resume behavior, checkpoint cadence, and evaluation path.
