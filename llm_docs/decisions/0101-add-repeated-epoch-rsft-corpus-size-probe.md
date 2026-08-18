---
status: accepted
date: 2026-08-18
---

# ADR 0101 — Add a repeated-epoch R-SFT corpus-size probe

## Context

The accepted atomic R0 checkpoint `100m-2b-rsft-r0-atomic-pilot-001` received only 29 optimizer updates over the frozen 630-example reasoning corpus plus the 10% S0 instruction-retention lane. In direct chat, even an in-distribution reasoning problem did not reliably emit the newly promoted `<think>` token as the first assistant token. The project owner wants a controlled diagnostic before expanding the corpus: hold the data, parent, token protocol, optimizer family, and training path fixed, but increase repeated exposure to the same immutable train stream.

This is explicitly an overfitting/corpus-size diagnostic, not a new claim about the preferred production training recipe.

## Decision

Add `--num-epochs N` to the Kaggle R-SFT launcher.

- `N=1` preserves the existing one-pass behavior.
- Repeat experiments run from the same completed S0 parent rather than continuing from the accepted 29-step R-SFT checkpoint.
- The immutable R-SFT train blocks are replayed in the same order for exactly `N` passes.
- Repeated passes use monotonically increasing logical block IDs, so checkpoint/resume remains exact across epoch boundaries.
- The WSD schedule spans the entire repeated stream rather than resetting every epoch.
- The bundle itself is not rewritten or duplicated.
- Canonical production `train` remains one pass. Repeated epochs are allowed only through the experimental/ablation lane.
- When `N>1` and no explicit run ID is supplied, the stable ID is `100m-2b-rsft-r0-{delimiter}-repeat-e{N}-001`.
- The requested 10-pass atomic probe therefore uses `100m-2b-rsft-r0-atomic-repeat-e10-001`.

With the current atomic bundle of 29 optimizer blocks, `--num-epochs 10` yields exactly 290 optimizer steps and ten exposures to every frozen training block.

## Purpose

The probe asks a narrow question: does much greater repeated exposure make the three new control tokens and the reasoning serialization behavior emerge reliably even without increasing corpus diversity? A strong improvement would indicate that the 29-update pilot was undertrained. Rapid memorization without general reasoning improvement would instead strengthen the case for a larger/diverse reasoning corpus.

## Non-decision

This does not replace the accepted atomic special-token decision, does not promote 10 epochs as a production default, and does not decide which external reasoning dataset will be mixed into the next larger R-SFT corpus.
