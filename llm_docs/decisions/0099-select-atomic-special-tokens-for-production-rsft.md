---
status: accepted
date: 2026-08-18
---

# ADR 0099 — Select atomic special tokens for production R-SFT

## Context

The first 100M/2B R0 delimiter ablation completed both matched one-pass Kaggle runs from the same completed S0 parent and the same shared R0/S0-retention source manifest.

The textual arm completed 30 optimizer steps / 58,099 loss-bearing train targets and reported validation loss 2.0444399 on 1,779 validation targets. The atomic arm completed 29 optimizer steps / 54,571 loss-bearing train targets and reported validation loss 2.4455797 on 1,653 validation targets. The atomic arm therefore had the higher teacher-forced validation loss in this deliberately tiny cold-start pilot.

The project owner explicitly decided that the production interface must nevertheless use dedicated special tokens. The reason is architectural and scientific rather than a claim that the atomic pilot won on validation loss: ordinary text such as the word `reasoning` carries natural-language semantics, while the model also needs an unambiguous machine-readable control concept for entering/exiting reasoning and entering the final answer. Reusing ordinary GPT-2 token sequences would entangle those roles and make output parsing depend on natural-language text. The project owner also notes that the atomic arm was learning three newly initialized, very frequent control rows during only one short pass, while its loss trajectory continued to decline at a similar qualitative rate.

## Decision

Production R-SFT uses only the frozen atomic reasoning-control interface:

```text
50257  <think>      reasoning start
50258  </think>     reasoning end
50259  <answer>     final-answer start
```

Consequences for the production contract:

- R-SFT production data must serialize reasoning with the three atomic IDs. Textual `Reasoning:` / `Answer:` delimiters are not a production option.
- The three token strings and IDs are semantic protocol, not ordinary content. Training/inference must preserve them atomically and output parsing may treat them as control boundaries.
- The textual 630-example run remains reproducible only as historical ablation evidence. It is not eligible as the parent or serialization contract for later production reasoning stages.
- The canonical production R0 run identity is `100m-2b-rsft-r0-001` once a production corpus is frozen.
- Kaggle production launch must fail closed if the supplied bundle declares `rsft.delimiter_format != atomic` or if its reasoning-token metadata does not exactly match the frozen token strings/IDs above.
- The production run remains one pass over its frozen bundle, preserving the previously accepted R-SFT scientific-control decision.
- The generic larger-run optimizer block remains 32,768 loss-bearing target tokens; the 2,048-target block was pilot-ablation-only.

## Deferred

This ADR does **not** freeze the production R0 corpus size/token budget or the production held-out allocation. Those remain separate data-scale decisions. It also does not freeze the later reasoning qualification suite.
