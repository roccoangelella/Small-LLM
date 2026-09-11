# MoE production launch contract

Date: 2026-09-11  
Status: Accepted; production wiring implemented on `moe-8e-top1`, exact GPU qualification pending

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

7. The Quantile-Balancing comprehension gate was completed on 2026-09-11. The accepted controller semantics are:
   - During all gradient-accumulation microbatches of one logical optimizer step, keep the currently committed selection bias fixed and accumulate the per-expert raw-score frontier needed for the exact `K/E` quantile.
   - On a successful optimizer step, compute each expert threshold `q_e` and replace the selection bias with `mean(q) - q_e`. Experts with a higher natural score threshold therefore receive a lower bias and become less likely to enter Top-K on the following step.
   - The balancing bias affects **Top-K selection only**. Mixture weights are computed from the selected experts' original unbiased scores.
   - If an optimizer attempt is skipped/retried because of overflow or another non-finite update, its transient score frontier is discarded and the previously committed bias remains unchanged. The failed attempt must have no persistent influence on later routing.

## Wiring status

Implemented on branch `moe-8e-top1` through head `d8bd2ca3880a10b05a133f75a32c2ad322c5fb47`:

- `MoEModelConfig.accepted()` now uses `semantic_vocab_size=8000`, `padded_vocab_size=8192`, 64 experts, Top-2, width-352 experts, sqrt-softplus routing, Quantile Balancing, and batched dropless expert GEMM.
- `python -m MOE_model --model-size accepted` is an explicit production identity. The generic balancing CLI no longer defaults the accepted path to `none`; attempts to downgrade accepted routing to `none` or `loss_free_sign` are rejected.
- Production telemetry reports the configured Top-K instead of the historical hard-coded Top-1 value.
- Dedicated provider-neutral production execution plus separate Modal H100 and Beam RTX4090 production launchers were added. Historical M0/M1 pilot launchers remain unchanged for reproducibility.
- Production commands pin GDN chunk size 32 so generic precision-dependent CLI defaults cannot silently mutate the accepted checkpoint-visible model configuration.
- Regression contracts now cover the 8,000/8,192 semantic-vs-physical vocabulary split, intrinsic Quantile selection, failed-attempt Quantile rollback semantics, production Top-K telemetry, fixed provider command identity, and the corpus-token bound (`7999` accepted; `8000` rejected).

No GitHub Actions workflow is attached to the branch head, and the exact 64E/Top-2 GPU qualification has **not yet been executed**. Therefore the wiring is implemented but is not yet authorized for the long production run. The accepted SuperBPE retokenization pipeline may be wired and qualified in parallel, but the long MoE training launch remains gated on the exact GPU qualification.
