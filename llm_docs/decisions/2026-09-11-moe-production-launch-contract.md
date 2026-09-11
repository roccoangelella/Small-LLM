# MoE production launch contract

Date: 2026-09-11  
Status: Accepted

## Decisions

1. The frozen SuperBPE tokenizer is the vocabulary source of truth for the production MoE model.
   - Exact semantic vocabulary: **8,000 token IDs** (`0..7999`).
   - Composition: **7,192 word-level byte-BPE entries + 800 SuperBPE entries + 8 special tokens = 8,000**.
   - `word_merges = 6,949` is the number of word-level merge operations, not the vocabulary size.
   - The model must therefore use `semantic_vocab_size = 8000`.
   - `padded_vocab_size = 8192` is retained solely as an internal tensor-alignment/storage optimization; IDs `8000..8191` are not semantic tokens and must not be exposed as output classes or accepted as corpus token IDs.

2. The MoE training entrypoint must explicitly support the accepted production model configuration (`MoEModelConfig.accepted()`), rather than routing production training through the legacy `substantive()` 8-expert/Top-1 pilot configuration.

3. **Quantile Balancing is part of the accepted production MoE model contract.** The accepted-model training path must enable Quantile Balancing intrinsically and must not silently replace or disable it through a generic CLI default. Experimental/smoke configurations may remain separately configurable.

4. The production provider launcher must launch the accepted **64-expert, Top-2, expert-width-352** model and not the legacy bounded 100M/2B M0/M1 pilot identity. Provider launch configuration must preserve the accepted model identity rather than reconstructing an older pilot configuration.

5. Before the long production run, the accepted path must pass a short exact-configuration qualification covering forward/backward/update, 64E/Top-2 routing, Quantile controller behavior, semantic-vocabulary bounds, checkpoint/resume, and production telemetry.

6. After the production wiring and qualification are ready, the next data operation is to **retokenize the full 100B pretraining corpus with the frozen `tokenizer/superbpe_8000.json` tokenizer**. GPT-2-tokenized corpus artifacts are not valid inputs for this MoE line.

7. Wiring the accepted code changes remains subject to the project comprehension gate: implementation proceeds only after the remaining Quantile-Balancing update semantics have been explained in the owner's own words.
