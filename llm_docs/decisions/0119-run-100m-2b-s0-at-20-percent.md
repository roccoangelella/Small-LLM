---
status: accepted
date: 2026-08-24
---

# ADR 0119: run a 20% 100M/2B S0 scaling experiment with unchanged stratification

## Decision

Build and privately publish a new S0 supervised-fine-tuning bundle for the completed 100M/2B pretrained parent at exactly 20% of the verified parent loss-bearing target count.

The completed parent consumed `2,001,000,448` targets, so the requested SFT train ceiling is:

```text
floor(2,001,000,448 × 0.20) = 400,200,089 loss-bearing targets
```

This is a separate scaling experiment from the frozen 4% S0 artifact. It must not overwrite or reuse the 4% run, dataset, W&B, checkpoint, bundle, or publication identities.

Keep the prior S0 data recipe unchanged except for total target horizon:

```text
85% filtered instruction targets
15% frozen original-distribution ClimbMix replay targets
```

Within the instruction portion, retain the same source stratification:

```text
75.0% smol-magpie-ultra-short
10.0% smol-contraints
 7.5% smollm-rewrite-30k
 7.5% smol-summarize-20k
```

Retain the pinned SmolTalk revision, identity-safe 95/2.5/2.5 split, filtering/decontamination policy, context/template contract, optimizer-block target, and deterministic seed used by the previous 4% S0 build. Do not introduce dataset repetition to satisfy the larger horizon. If the unique filtered instruction sources are exhausted, the build must fail closed and that exhaustion becomes evidence for the next dataset-design decision.

## CLI implementation

The canonical `kaggle/launch_sft.py` surface now accepts `--sft-fraction` on prepare, publish, train, and eval. Accepted forms include percentages (`20%`), decimals (`0.20`), and ratios (`1/5`). A non-default fraction derives separate experiment identities and separate 100M work/bundle paths while leaving the historical 4% profile unchanged.

For the 100M/2B 20% experiment the derived identities are:

```text
SFT run / W&B run: 100m-2b-sft-s0-20pct-001
Kaggle dataset slug: small-llm-100m-2b-sft-s0-20pct-001
```

The lower-level scaled builder already takes an explicit rational fraction and preserves the existing `0.85/0.15` mixture and default instruction-source shares, so the new CLI flag exposes that existing capability rather than defining a new dataset recipe.

## Rationale

The completed 4% S0 run retained most base-language capability but only weakly established strict instruction behavior. A substantially larger S0 target horizon is therefore useful as a controlled data-scaling experiment before attributing the limitation to model capacity or to the S0 mixture itself. Holding stratification and preprocessing fixed isolates total SFT exposure as the primary experimental variable.

## Qualification

The 20% artifact is experimental until it passes the same frozen parent-versus-SFT qualification matrix. In particular, compare instruction behavior, EOS/runaway behavior, general `eval_core_v1` retention, and any later reasoning/generalization probes. A lower SFT-distribution validation loss alone is not sufficient for promotion.
