---
status: accepted
date: 2026-08-18
supersedes: 0094
---

# 0095 — Decay from 1e-4 to 1e-5, then finish at 5e-6

## Context

ADR 0094 already lowered the post-step-15,500 continuation to a `1e-4` settling endpoint, but its inverse-square-root middle would still reach only about `4.93e-5` at the 9.6B-token terminal-cooldown boundary. The user judged that learning-rate scale still too high given the repeated project evidence that validation improves while LR is falling and degrades or stalls when the decay becomes much gentler.

The user therefore authorized explicit long-phase endpoints: reach `1e-4` after the short settling phase, decay from `1e-4` to `1e-5` across the long middle of the run, then decay from `1e-5` to `5e-6` over the final approximately 400M targets.

Recent under-1B continuation-schedule work generally favors decreasing power-law / inverse-square-root-like LR rather than a long flat WSD plateau. A literal inverse-square-root base cannot satisfy the newly requested `1e-4 -> 1e-5` endpoint pair over this token interval. To retain a smooth power-law continuation while obeying the project evidence and requested endpoints, calibrate the power exponent from the two endpoint constraints instead of choosing it arbitrarily. This gives an exponent of approximately `1.6270515945`, materially steeper than the usual approximately `0.5` continuation exponent; this is an intentional project-specific experiment, not a claim that such a steep exponent is generally state of the art.

## Decision

Supersede ADR 0094. The authorized main 100M/10B continuation must always fork the exact original uncooled `100m-10b-data-001/checkpoints/step-00015500` state. Do not continue from ADR-0093 or ADR-0094 branches and do not reheat a cooled checkpoint.

Preserve model, optimizer, scaler, RNG, data cursor, architecture, hybrid Muon+AdamW recipe, FP16 precision, microbatch 4, frozen validation prefix, and exact 10B corpus order. Change only the LR scheduler.

Use three phases:

```text
source step:                 15,500
source targets:              2,031,616,000
source LR:                   3.0e-4

phase 1:                     cosine settle
settle span:                 300,023,808 targets / 2,289 updates
settle end step:             17,789
settle end targets:          2,331,639,808
settle end LR:               1.0e-4

phase 2:                     calibrated power-law decay
formula:                     1.0e-4 * (2,331,639,808 / committed_targets)^p
p:                           ~1.6270515945
cooldown start step:         73,242
cooldown start targets:      9,599,975,424
LR at cooldown start:        1.0e-5

phase 3:                     linear terminal cooldown
cooldown span:               400,031,744 targets / 3,052 updates
cooldown start LR:           1.0e-5
final LR:                    5.0e-6
final step:                  76,294
final targets:               10,000,007,168
```

Approximate phase-2 LR landmarks are:

```text
step 20,000:  8.26e-5
step 25,000:  5.75e-5
step 30,000:  4.27e-5
step 40,000:  2.68e-5
step 50,000:  1.86e-5
step 60,000:  1.38e-5
step 70,000:  1.08e-5
step 73,242:  1.00e-5
step 76,294:  5.00e-6
```

Extend the generic `wsqd` trainer schedule with a positive `base_power` parameter whose default remains `0.5`. Omit the default field from serialized historical WSD/WSqD configurations so existing checkpoint identities remain compatible. The deep-decay launcher sets the calibrated exponent explicitly.

Use `beam/deep_decay_10b_from_15500.py` with run ID `100m-10b-deep-decay-from-step15500`. Keep its local, W&B, and Hugging Face checkpoint namespaces separate from all earlier continuations and diagnostics.

## Consequences

- ADR 0094 becomes historical schedule evidence rather than the authorized main continuation.
- The long middle is now far more aggressive than inverse-square-root decay and is explicitly constrained to hit `1e-5` before the terminal cooldown.
- The final 400M targets make only a small final convergence move, `1e-5 -> 5e-6`, rather than carrying the bulk of the LR reduction.
- The schedule remains continuous at both phase boundaries and preserves exact checkpoint/resume semantics through the trainer scheduler.
- This is intentionally more aggressive than current generic continuation-schedule defaults; validation behavior must be watched closely for under-training or premature LR collapse.

## Links

- [`0094-lower-post15500-lr-scale-throughout-10b.md`](0094-lower-post15500-lr-scale-throughout-10b.md)
- [`0093-front-load-10b-decay-and-use-1p5e-5-terminal-lr.md`](0093-front-load-10b-decay-and-use-1p5e-5-terminal-lr.md)
- [`0091-use-step15500-for-controlled-400m-cooldown-probe.md`](0091-use-step15500-for-controlled-400m-cooldown-probe.md)
- [`../runbooks/100m_10b_beam.md`](../runbooks/100m_10b_beam.md)
