# MoE production launch contract

Date: 2026-09-11  
Status: Accepted

## Decisions

1. The frozen SuperBPE tokenizer is the vocabulary source of truth for the production MoE model.
   - Exact semantic vocabulary: **8,000 token IDs** (`0..7999`).
   - Composition: **7,192 word-level byte-BPE entries + 800 SuperBPE entries + 8 special tokens = 8,000**.
   - `word_merges = 6,949` is the number of word-level merge operations, not the vocabulary size.
   - The model must therefore use `semantic_vocab_size = 8000`.
   - `padded_vocab_size = 8192` may be retained solely as an internal tensor-alignment/storage optimization; IDs `8000..8191` are not semantic tokens and must not be exposed as output classes or accepted as corpus token IDs.

2. The MoE training entrypoint must explicitly support the accepted production model configuration (`MoEModelConfig.accepted()`), rather than routing production training through the legacy `substantive()` 8-expert/Top-1 pilot configuration.

3. **Quantile Balancing is part of the accepted production MoE model contract.** The accepted-model training path must enable Quantile Balancing and must not silently replace or disable it through a generic CLI default. Experimental/smoke configurations may remain separately configurable.

4. Wiring these accepted code changes is pending the project comprehension gate: the implementation will be changed only after the user explains the relevant model-selection, vocabulary, and Quantile-Balancing semantics in their own words.
