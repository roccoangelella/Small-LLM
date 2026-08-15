---
status: accepted
date: 2026-08-15
---

# ADR 0089 — Require explicit chat stage and R-SFT tokenizer selection

## Context

R-SFT introduces three semantic control-token IDs on rows 50,257–50,259 while pretrained and S0 SFT checkpoints retain the ordinary 50,257-token GPT-2 semantic vocabulary. The local chat CLI must not guess which tokenizer contract belongs to a selected model artifact.

Before this decision, `chat.py` treated pretrained selection as the default and used `--sft` only as an override. That becomes unsafe once R-SFT checkpoints can emit and consume the promoted reasoning-token IDs.

## Decision

The local `chat.py` stage is now mandatory and explicit. Every invocation must select exactly one of:

- `--pre-trained`
- `--sft`
- `--r-sft`

`--pre-trained` and `--sft` use the unchanged normal GPT-2 tokenizer contract and require `semantic_vocab_size=50_257`.

`--r-sft` uses the R-SFT tokenizer extension and requires `semantic_vocab_size=50_260`. IDs 50,257, 50,258, and 50,259 are the fixed reasoning-start, reasoning-end, and answer-start slots respectively.

The exact marker strings remain configurable and are not frozen by this ADR. A completed R-SFT checkpoint must carry the three marker spellings and fixed IDs in verified pipeline metadata under `pipeline_state.reasoning_tokenizer`. Chat reconstructs the extended tokenizer from that artifact metadata and fails closed if the mapping is missing, malformed, or disagrees with the model vocabulary.

The base S0 tokenizer is not modified globally. The R-SFT encoder wraps GPT-2 only for R-SFT data/inference, so pretrained and S0 tokenization remain byte/token compatible with their existing artifacts.

R-SFT chat run IDs are not invented in advance. `--r-sft` resolves only explicitly registered completed R-SFT artifacts; until such an artifact is frozen and registered, that stage reports that no profile is registered.

## Consequences

- A user can no longer accidentally load the pretrained 100M/2B artifact merely by omitting `--sft`; a stage flag is required.
- An R-SFT checkpoint cannot be decoded with the ordinary GPT-2 tokenizer through the supported chat path.
- Generated reasoning-marker IDs can be streamed, decoded, stored in chat history, and re-encoded atomically on later turns.
- The textual-versus-atomic serialization ablation remains intact because token spellings are still artifact-provided rather than hardcoded here.
- When the first completed R-SFT run identity is frozen, it must be added to the explicit R-SFT chat registry and its checkpoint writer must persist `ReasoningTokenSpec.to_metadata()` in pipeline state.
