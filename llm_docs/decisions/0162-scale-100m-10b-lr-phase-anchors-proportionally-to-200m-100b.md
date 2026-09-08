---
status: accepted
date: 2026-09-08
supersedes: null
owners: [Small-LLM]
---

# ADR 0162: scale the successful 100M/10B LR phase anchors proportionally to 200M/100B

## Context and problem statement

ADR 0160 selects approximately 200M parameters / 100B target tokens as the next major pretraining trajectory. ADR 0161 keeps the successful 100M/10B deep-decay / WSqD-style LR structure while modestly broadening the numerical LR range at each major anchor.

The remaining timing question is where those LR anchors should occur over a horizon that is ten times longer than the completed 100M/10B run.

The successful 100M/10B trajectory reached the deep-decay source anchor at step 15,500 / 2,031,616,000 targets, completed its aggressive settle at step 17,789 / 2,331,639,808 targets, began terminal cooldown at step 73,242 / 9,599,975,424 targets, and finished at step 76,294 / 10,000,007,168 targets. The inherited fresh-run WSD warmup occupied approximately the first 5% of the total horizon.

## Decision outcome

Chosen option: **preserve the same phase positions as fractions of the successful 100M/10B trajectory and scale the token/update anchors by 10x for the 200M/100B run.**

With the unchanged block-64, context-2048 global update geometry, one optimizer update represents 131,072 target tokens. The exact block-aligned 100B endpoint is therefore 100,000,071,680 targets at step 762,940.

The proportional timing contract is:

```text
fresh warmup endpoint:        step  38,147   targets   5,000,003,584   (~5.0000%)
deep-decay start / peak:      step 155,000   targets  20,316,160,000   (~20.3161%)
aggressive-settle endpoint:   step 177,890   targets  23,316,398,080   (~23.3164%)
terminal-cooldown start:       step 732,420   targets  95,999,754,240   (~95.9997%)
final endpoint:                step 762,940   targets 100,000,071,680   (100%)
```

Therefore the 200M/100B schedule preserves the same relative chronology as the successful 10B trajectory:

1. warm up to the selected widened peak over approximately the first 5B targets;
2. remain at the peak through the proportional deep-decay-start anchor at approximately 20.316B targets;
3. perform the aggressive settle from approximately 20.316B to 23.316B targets;
4. perform the long calibrated decay from approximately 23.316B to 95.9998B targets;
5. perform terminal cooldown over the final approximately 4.0003B targets.

ADR 0161 still governs the numerical LR direction: the 200M/100B LR anchors are to be modestly broader than the 100M/10B values. This ADR freezes timing only; it does not independently freeze the exact widened LR values or the new power-law exponent required to connect them.

## Consequences

### Positive

- Phase timing is a direct proportional extrapolation of the project's best-tested 100M/10B schedule rather than an arbitrary redesign.
- The exact block-64 step boundaries are simple 10x multiples of the successful deep-decay anchors.
- The long central decay receives approximately 72.68B targets, preserving the successful schedule's relative allocation while exploiting the longer horizon.
- Exact-resume and scheduler-state validation can use deterministic integer update boundaries.

### Negative or limiting

- The proportional extrapolation includes a long peak-LR interval from the end of warmup to approximately 20.316B targets; this is a deliberate consequence of preserving the original schedule's chronology and must be watched closely in early validation.
- The exact widened LR values and connecting power exponent remain to be frozen.
- The 200M geometry may respond differently from 100M, so the proportional schedule remains an extrapolation that requires live stability and validation monitoring.

## Validation

Before production launch:

1. freeze the exact widened LR anchors from ADR 0161;
2. solve the continuous decay function/exponent across the fixed anchors above;
3. verify exact step/token arithmetic against the trainer planner;
4. run short 200M probes covering warmup/peak stability and the beginning of the aggressive settle;
5. confirm scheduler state survives exact checkpoint/resume without phase drift.

## Links

- `llm_docs/decisions/0160-select-200m-100b-as-next-pretraining-run.md`
- `llm_docs/decisions/0161-broaden-200m-100b-lr-anchor-range-slightly.md`
- `llm_docs/decisions/0095-decay-1e-4-to-1e-5-then-5e-6.md`
- `llm_docs/evidence/scaling/100m_10b_step17789_decay_transition_learning_efficiency_2026-08-21.md`
