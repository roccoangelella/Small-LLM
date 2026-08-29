---
status: accepted
date: 2026-08-25
supersedes: 0119-run-100m-2b-s0-at-20-percent
---

# ADR 0122: replace the 20% S0 trial with a 10% capacity-aware S0 mixture

## Decision

Do not run the previously authorized 20% 100M/2B S0 scaling experiment. The next S0 trial on the completed `100m-2b-data-001` parent will instead use exactly 10% of the verified parent loss-bearing target count.

The parent consumed `2,001,000,448` targets, so the requested train ceiling is:

```text
floor(2,001,000,448 × 0.10) = 200,100,044 loss-bearing targets
```

The run remains a distinct experiment from the completed 4% S0 artifact. Its intended identities are:

```text
SFT run / W&B run: 100m-2b-sft-s0-10pct-001
Kaggle dataset slug: small-llm-100m-2b-sft-s0-10pct-001
```

No instruction example is repeated merely to reach the horizon.

## Top-level mixture

Preserve the scientifically important top-level S0 mixture:

```text
85% unique filtered instruction targets
15% frozen original-distribution ClimbMix replay targets
```

Using integer target counts at the 200,100,044-target ceiling, the normative planning counts are:

```text
instruction: 170,085,037 targets
replay:       30,015,007 targets
```

The one-target rounding remainder is assigned to replay so the exact counts sum to the requested train ceiling.

## Capacity-aware internal instruction stratification

The previous nominal instruction-source weights (`75/10/7.5/7.5`) cannot be preserved at this horizon without repeating finite sources. The audited train pools expose the following complete unique capacities for the three smaller sources:

```text
smol-contraints:       4,026,530 targets
smollm-rewrite-30k:    3,762,301 targets
smol-summarize-20k:    1,588,795 targets
```

Use all of those unique targets, then fill the remaining instruction budget from `smol-magpie-ultra-short`. This produces the following planned instruction allocation:

```text
smol-magpie-ultra-short: 160,707,411 targets
smol-contraints:            4,026,530 targets
smollm-rewrite-30k:         3,762,301 targets
smol-summarize-20k:         1,588,795 targets
                           -----------
total instruction:        170,085,037 targets
```

Expressed within the instruction lane, the intended source weights are approximately:

```text
94.4865% smol-magpie-ultra-short
 2.3674% smol-contraints
 2.2120% smollm-rewrite-30k
 0.9341% smol-summarize-20k
```

Across the complete SFT train stream, the corresponding planned shares are approximately:

```text
80.3135% smol-magpie-ultra-short
 2.0123% smol-contraints
 1.8802% smollm-rewrite-30k
 0.7940% smol-summarize-20k
15.0000% ClimbMix replay
```

The exact target counts above are normative; percentages are descriptive and may differ by tiny complete-record packing/rounding effects. The builder must not leave the old nominal source weights active and simply allow sources to exhaust, because that would renormalize the surviving Magpie/replay lanes implicitly and push replay above the intended 15%.

## Held-out policy

Do not scale validation or test with the 10% training horizon. Reuse the already-frozen S0 validation/test splits from the completed 4% 100M/2B S0 bundle so the SFT scaling comparison is evaluated on identical held-out examples.

This replaces the current proportional held-out sizing behavior for this scaling trial. A larger train horizon is not a reason to enlarge an already adequate fixed held-out set, and changing held-out examples would weaken direct comparison against the 4% S0 checkpoint.

## Builder requirements

Before publication, the 10% bundle builder must verify all of the following:

- requested train horizon is `200,100,044` loss-bearing targets, subject only to the existing complete-record shortfall tolerance;
- top-level realized instruction/replay mix remains 85/15 within packing tolerance;
- no instruction-record repetition is introduced;
- the three small instruction sources are consumed up to their audited unique capacities above, subject only to final record-fit effects;
- Magpie supplies the remaining instruction budget rather than replay absorbing exhausted-source quota;
- validation and test identities match the frozen completed-4% S0 held-out splits;
- the build report records realized per-source target counts and percentages explicitly.

If Magpie cannot supply the planned `160,707,411` instruction targets after filtering and serialization, fail closed rather than changing the mixture silently.

## Rationale

The attempted 20% configuration exposed two independent finite-data constraints. First, proportional validation/test scaling requested roughly 10.53M targets per held-out split even though the unique filtered pools contain only roughly 5.27M validation and 5.11M test targets; fixed held-outs are sufficient and superior for longitudinal comparability. Second, 20% SFT at an 85% instruction share would require roughly 340.17M instruction targets, far beyond the unique S0 instruction inventory.

A 10% horizon requires only 170.09M instruction targets and therefore fits comfortably inside the estimated unique Magpie capacity while retaining the full smaller-source inventories. It gives approximately 2.5× the SFT exposure of the completed 4% run without introducing instruction repetition, while preserving the 85/15 instruction-versus-pretraining-replay control.

## Qualification

This 10% artifact remains experimental until it passes the same frozen parent-versus-SFT qualification matrix used for the completed 4% S0 run: instruction behavior, EOS/runaway/repetition behavior, held-out SFT objective, unchanged `eval_core_v1` retention, and subsequent reasoning/generalization probes where applicable. Lower SFT-distribution validation loss alone is not sufficient for promotion.
