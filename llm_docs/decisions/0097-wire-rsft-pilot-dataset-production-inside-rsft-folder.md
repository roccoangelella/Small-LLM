---
status: accepted
date: 2026-08-18
---

# ADR 0097 — Wire R-SFT pilot dataset production inside the R-SFT folder

## Context and problem statement

The R0 teacher prompts, strict teacher schema, 630-example pilot size, 90/10 reasoning-retention policy, S0 retention-source policy, reasoning-tokenizer contract, S0-to-R-SFT model transition, and dedicated Kaggle 2xT4 training entry point are already defined. The missing data-side step is to turn live Gemini output plus a completed S0 bundle into the exact immutable datasets consumed by the R-SFT trainer.

The project owner requested that this production path live inside `post_training/R-SFT/` rather than creating another top-level dataset or Kaggle-specific data subsystem.

## Decision outcome

- Keep R-SFT dataset production inside `post_training/R-SFT/`.
- Provide one human entry point, `post_training/R-SFT/produce.py`, backed by reusable production/bundle modules in the same folder.
- The first pilot generation remains 30 examples in each of 21 skill x difficulty cells, or 630 total examples / 63 default Gemini calls.
- Save every schema-valid Gemini batch independently so interrupted live generation is resumable without repeating already-valid API calls.
- Build the textual and atomic delimiter ablation bundles from the exact same frozen reasoning examples, deterministic train/held-out partition, retained S0 record identities, and semantic record order.
- Sample the 10% retention lane from the exact tokenized instruction records in the completed S0 bundle, preserving the instruction-source shares recorded by that bundle. Do not include S0 ClimbMix replay in R-SFT retention.
- Emit native `small-llm-sft-bundle` train/validation/test shards so the already-qualified SFT reader, verifier, trainer, checkpointing, and Kaggle 2xT4 runtime remain reusable.
- Carry the artifact-provided three-token spelling into `reasoning-tokens.json`; dataset production does not hardcode or newly freeze the three atomic marker strings.
- Continue to trust Gemini semantics for this pilot. Production rejects malformed/schema-drifting data and mechanical dataset inconsistencies, but does not add a new symbolic or LLM semantic verifier.

## Implementation consequences

- `production.py` owns resumable generation/orchestration and matched-root verification.
- `bundle.py` owns deterministic partitioning, exact S0 tokenized-retention reuse, arm-specific reasoning serialization, native SFT shard writing, and matched pilot identities.
- `schema.py` can strictly reload the frozen reasoning JSONL.
- `serialization.py` can serialize reasoning examples into train/validation/test splits.
- The 30-example pilot uses one validation and one test example per skill x difficulty cell, leaving 28 train examples per cell. This keeps all 21 cells represented in both held-out splits while preserving an identical partition between ablation arms.
- Textual and atomic delimiter representations necessarily have slightly different target-token counts. To avoid changing retained examples between arms, the pilot chooses one symmetric retention target from their mean reasoning-target count, then records the realized retention share separately for each arm.

## Consequences

- One command can generate and freeze the live pilot, then materialize both Kaggle-ready delimiter arms.
- Generation can also be run independently before the exact atomic marker spellings are selected.
- Retention is guaranteed to be data the S0 model actually trained on rather than a fresh resample from upstream SmolTalk.
- The delimiter ablation remains substantially cleaner because reasoning content, retention identities, split membership, and record ordering are held fixed.
- Final production R-SFT corpus size remains open; this wiring is for the already-approved 630-example pilot and its delimiter ablation.
