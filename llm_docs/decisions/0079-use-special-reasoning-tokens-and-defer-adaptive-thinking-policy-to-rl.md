---
status: accepted
date: 2026-08-14
---

# ADR 0079 — Use special reasoning-control tokens and defer adaptive thinking policy to RL

## Context and problem statement

Reasoning SFT needs a parseable boundary between hidden work and final answers, while the small student should not be forced to reason at a fixed length on every prompt.

## Considered options

1. Use ordinary textual delimiters only.
2. Force an unconditional reasoning mode and optimize directly for short traces.
3. Target atomic control tokens, validate them against textual delimiters, and defer adaptive entry/length policy to RL/RLVR.

## Decision outcome

For the first reasoning-oriented post-training lane after instruction SFT:

- Target a three-token atomic reasoning-control interface. The exact token strings and final role of each marker remain to be frozen with the reasoning serialization contract; the implementation must not silently invent those semantics.
- The model should be allowed to answer directly on simple prompts or enter an explicit reasoning mode when useful; reasoning is not mandatory for every prompt.
- Reasoning SFT is responsible for cold-starting valid reasoning structure and concise, correct traces. A later RL/RLVR stage is responsible for learning when reasoning is useful and how much reasoning budget to spend.
- Before the production reasoning run, perform a bounded serialization ablation comparing ordinary textual delimiters (for example `Reasoning:` / `Answer:`) against atomic special-token delimiters. The expected production direction is special tokens, but the ablation remains required evidence.
- Do not use a naive unconditional inverse-length reward as the default RL objective. Correctness/verifiability remains primary; any efficiency pressure should be correctness-gated, tolerance-banded, or adaptively weighted so that reasoning acquisition and exploration are not suppressed.

## Tokenizer / model feasibility

The current model has `semantic_vocab_size=50_257` and `padded_vocab_size=50_304`, leaving 47 physically allocated but non-semantic embedding/output rows. Therefore three new atomic tokens can be assigned to existing padded rows (for example IDs 50_257–50_259) by increasing the semantic vocabulary to 50_260 while keeping the padded vocabulary and embedding matrix shape unchanged.

Those rows were not semantically pretrained. They must be initialized deliberately and trained during reasoning SFT. Because the input embedding and LM projection are tied, the same rows learn both as input control markers and as output tokens predicted by the model.

The promoted reasoning-token rows will use the same frozen token-embedding initialization policy used by the pretrained model rather than a custom semantic-average initializer. Under the selected normal initializer this is the existing GPT-style `Normal(0, 0.02)` matrix initialization, applied only to the newly promoted rows under the run's deterministic seed policy. Existing pretrained embedding rows are preserved unchanged; the remaining padded rows stay non-semantic/zeroed.

## Rationale

The control-token interface makes reasoning boundaries machine-parseable for later verifiers and RL while preserving the option to bypass reasoning on trivial prompts. Recent 2026 efficient-reasoning work warns that fixed or unconditional length penalties can cause under-thinking or suppress useful exploration, whereas correctness-gated and adaptive penalties preserve accuracy more reliably.

Using the original embedding initializer for promoted control-token rows keeps their cold-start treatment consistent with the model's established initialization contract and avoids introducing an unrelated token-specific heuristic into the reasoning-format ablation.

## Follow-up

1. Freeze the reasoning skill taxonomy and L1/L2/L3 task definitions.
2. Freeze the exact three-token serialization semantics.
3. Implement tokenizer/model support using the existing padded rows and the frozen embedding initializer for promoted rows.
4. Run the textual-delimiter vs atomic-token pilot.
5. Build the verified R-SFT dataset.
6. After R-SFT qualification, design the adaptive RLVR reward for correctness plus reasoning efficiency.

## Consequences

- The production reasoning format remains gated on a textual-versus-atomic delimiter ablation.
- Promoted padded rows preserve the pretrained embedding shape but require deterministic initialization and training.
- Adaptive reasoning length and entry policy are deferred to later RL/RLVR rather than forced during initial SFT.
