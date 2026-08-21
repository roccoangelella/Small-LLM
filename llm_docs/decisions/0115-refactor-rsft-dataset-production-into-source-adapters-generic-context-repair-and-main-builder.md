---
status: accepted
date: 2026-08-21
supersedes: null
---

# ADR 0115 — Refactor R-SFT dataset production into source adapters, generic context repair, and one main builder

## Context and problem statement

The R-SFT dataset directory accumulated separate Superior-Reasoning scripts for the original Stage-1 production corpus, Stage-1 over-context adaptation/resume, candidate review, and the new Stage-2 scaling lane. That organization couples generic operations such as 2,048-token repair and GemRouter retries to one upstream dataset and would multiply one-off scripts when additional reasoning datasets are added for larger runs.

The current scaling plan also needs Stage 1 and Stage 2 of Superior Reasoning to behave as one source family with one normalization/filter contract, while keeping the completed 16,716-row Stage-1-expanded artifact immutable as the base of the 1% experiment.

## Decision outcome

Adopt a three-layer R-SFT dataset-production architecture:

1. **Source adapters.** Each upstream reasoning dataset owns one adapter that streams its native data and maps it into the common R-SFT candidate contract. Superior Reasoning gets one active source adapter covering both `stage1` and `stage2`. Stage-specific files are not part of the active architecture.
2. **Generic over-context adaptation.** Curation, the GemRouter/Gemini transport gate, Variant-D compression, retries/split recovery, exact 2,048-token revalidation, resumability, and adapted-JSONL finalization live in one source-agnostic module. Future source datasets reuse it without copying Superior-specific adaptation code.
3. **One main builder.** A single R-SFT dataset entry point prepares registered sources, aggregates/deduplicates their fit and over-context streams, optionally consumes generic adapted rows, and assembles the final reasoning JSONL at a loss-bearing token budget.

For now the source registry contains only Superior Reasoning. Adding a future source should require a new source adapter and registry entry, not another GemRouter implementation or top-level dataset-building script.

## Superior Reasoning policy

The active Superior adapter covers both Stage 1 and Stage 2 and keeps the accepted R0 policy unchanged:

- `instruction_following` only;
- strict `<think>...</think>` parsing and non-empty answer;
- normalized-prompt deduplication;
- primary math/computation and primary code/programming exclusions;
- reserved reasoning-marker rejection;
- exact atomic 2,048-token serialization with no truncation;
- context-fit rows enter directly and otherwise-clean over-context rows are emitted to the generic repair lane.

For the current 1% experiment, the frozen 16,716-row Stage-1-expanded JSONL remains the base and the main builder defaults to adding Superior Stage 2. Reprocessing Stage 1 is therefore unnecessary for the current run, although the unified adapter can stream either or both stages for future clean rebuilds.

## Generic GemRouter policy

Move the fidelity-first compression prompt out of the Superior source policy and treat it as the default R-SFT over-context repair prompt. **Its text is unchanged for this refactor.** A future ADR may select source-specific compression prompts, but the current implementation deliberately does not vary them.

The generic repair lane preserves the existing constraints:

- at most four examples per request;
- explicit `keep / exclude_math / exclude_code / exclude_safety` curation before teacher traffic;
- Gemini-only GemRouter health gate with `backendOrder=["gemini-api"]` and `fallbackEnabled=false`;
- strict JSON response shape and ID order;
- fidelity-first retry correction and deterministic split recovery;
- exact post-rewrite atomic context validation;
- no truncation, silent repair, alternate provider, or replay-as-volume shortcut.

## Compatibility

Historical Superior-specific scripts and frozen artifacts remain available for reproduction of the already-completed Stage-1 trajectories. They are legacy compatibility paths, not the preferred interface for new corpus creation. New work should enter through `post_training/R-SFT/dataset/build.py` and the generic `over_context.py` module.

## Consequences

- Stage 1 and Stage 2 are represented as one Superior Reasoning source family.
- The current 1% build can first try context-fit Stage-2 rows and only enter GemRouter repair if the token target is still short.
- Future reasoning datasets can be added without cloning the over-context adaptation machinery.
- Dataset selection, repair, and final assembly have explicit boundaries, making scaling experiments easier to audit and test.
- The existing 90/10 reasoning/S0-retention bundle contract and downstream atomic bundle builder are unchanged.
