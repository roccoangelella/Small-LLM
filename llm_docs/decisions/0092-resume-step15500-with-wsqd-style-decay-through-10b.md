---
status: accepted
date: 2026-08-17
supersedes: 0057, 0071
---

# 0092 — Resume step 15,500 with WSqD-style decay through the 10B endpoint

## Context

The controlled cooldown fork from `step-00015500` began outperforming the original flat-LR 100M/10B trajectory within roughly the first 500 cooldown updates. The user therefore authorized a replacement long-horizon trajectory that resumes from the exact uncooled step-15,500 state and uses a continuously decreasing base LR for the rest of the 10B corpus instead of returning to the original long `3e-4` WSD stable phase.

Recent 2026 schedule work motivates this direction. WSqD replaces WSD's long constant stable phase with a shifted/inverse-square-root base and keeps a terminal linear cooldown so continuation anchors remain useful without fixing a high LR for the whole horizon. For this project we use that structure while retaining the existing explicit `0.1` minimum-LR floor rather than claiming a literal reproduction of the paper.

The exact step-15,500 source has consumed `2,031,616,000` targets and still has the original model, optimizer, scaler, RNG, and data cursor. The exact 10B dataset endpoint is block-aligned at step `76,294` / `10,000,007,168` targets.

## Decision

Create a new main continuation branch, `100m-10b-wsqd-from-step15500`, from exact checkpoint `100m-10b-data-001/checkpoints/step-00015500`.

Preserve model parameters, optimizer state, scaler state, RNG state, data cursor, model architecture, hybrid Muon+AdamW recipe, FP16 precision, microbatch 4, frozen validation prefix, and exact 10B corpus order. Change only the LR scheduler.

Use an anchored inverse-square-root base from the source checkpoint:

```text
anchor step:                 15,500
anchor targets:              2,031,616,000
anchor LR:                   3e-4
base schedule:               LR(t) = 3e-4 * sqrt(2,031,616,000 / t)
terminal cooldown start:     step 73,242
cooldown-start targets:      9,599,975,424
LR at cooldown start:        ~1.38009e-4
terminal cooldown:           linear
cooldown updates:            3,052
cooldown targets:            400,031,744
minimum LR ratio:            0.1
final LR:                    3e-5
final step:                  76,294
final targets:               10,000,007,168
```

The terminal cooldown linearly interpolates from the inverse-square-root base LR at step 73,242 to `3e-5` at the exact 10B endpoint. This is intentionally described as **WSqD-style / WSqD-inspired** because the project retains a nonzero terminal floor.

Implement a generic trainer schedule kind `wsqd` with explicit `schedule_anchor_tokens` and `cooldown_start_tokens`, while preserving serialized config identity for historical constant/WSD checkpoints by omitting the new zero-valued fields outside WSqD.

Use `beam/wsqd_10b_from_15500.py` as the launcher. It must fail closed unless the exact original step-15,500 source is present, fork that state into a separate run namespace, CPU-stage and verify the checkpoint-aligned dataset window before GPU allocation, and publish W&B/Hugging Face checkpoints under the new run ID.

The ongoing 400M cooldown probe remains a diagnostic and should finish for comparison, but its cooled endpoint is not the parent of this long trajectory. The new 10B continuation always forks the original uncooled step-15,500 checkpoint.

## Consequences

- The original flat-`3e-4` stable phase through most of 10B is no longer the authorized main trajectory.
- The new run consumes the remaining `60,794` optimizer blocks from step 15,500 through the exact 10B endpoint.
- LR decreases immediately after the fork rather than waiting for a late WSD cooldown.
- The final approximately 400M targets are a committed terminal cooldown; extending beyond 10B would require resuming from a pre-cooldown WSqD checkpoint rather than reheating the final model.
- The completed 400M diagnostic probe and the long continuation must retain separate run/checkpoint namespaces.

## Links

- [`0091-use-step15500-for-controlled-400m-cooldown-probe.md`](0091-use-step15500-for-controlled-400m-cooldown-probe.md)
- [`0057-use-standard-wsd-for-100m-10b.md`](0057-use-standard-wsd-for-100m-10b.md)
- [`../runbooks/100m_10b_beam.md`](../runbooks/100m_10b_beam.md)
