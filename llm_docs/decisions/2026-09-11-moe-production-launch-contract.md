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

8. The 100B MoE corpus reuses the existing deterministic ClimbMix stratification/sharding/Hugging Face durability pipeline, but retokenization happens **before token-count scheduling** at the document boundary:
   - Read each original ClimbMix JSONL record as GPT-2 token IDs plus its existing `cluster_id` and stable source identity.
   - Remove a terminal GPT-2 EOD token when present, decode the document through the canonical GPT-2 byte-level tokenizer, then encode the resulting text with the frozen `tokenizer/superbpe_8000.json` artifact.
   - Build `SourceDocument` from the **SuperBPE IDs** while preserving the original cluster, source identity and deterministic train/validation split. Therefore deficit scheduling, rolling mixture checks, queue accounting and corpus stopping limits are all measured in the model's SuperBPE-token unit rather than inherited GPT-2 token counts.
   - The stream contract for this corpus uses semantic vocabulary size **8,000** and SuperBPE `<|endoftext|>` ID **7992**. GPT-2 EOD `50256` must never be emitted into the final shards.
   - Existing context+1 packing, immutable uint16 schema-v2 shards, crash-safe resume, READY frontier and Hugging Face upload/durability logic are retained.
   - The new tokenizer identity/hash, semantic vocabulary and EOD identity are part of production dataset configuration/schema identity so GPT-2 and SuperBPE corpora cannot be mistaken for one another or resumed across configurations.
   - The target corpus size is counted directly in **SuperBPE source tokens**; it is not inferred from the old GPT-2 token count.

## Wiring status

MoE model production wiring was implemented on branch `moe-8e-top1` through `d8bd2ca3880a10b05a133f75a32c2ad322c5fb47`:

- `MoEModelConfig.accepted()` uses `semantic_vocab_size=8000`, `padded_vocab_size=8192`, 64 experts, Top-2, width-352 experts, sqrt-softplus routing, Quantile Balancing, and batched dropless expert GEMM.
- `python -m MOE_model --model-size accepted` is an explicit production identity. The generic balancing CLI no longer defaults the accepted path to `none`; attempts to downgrade accepted routing to `none` or `loss_free_sign` are rejected.
- Production telemetry reports the configured Top-K instead of the historical hard-coded Top-1 value.
- Dedicated provider-neutral production execution plus separate Modal H100 and Beam RTX4090 production launchers were added. Historical M0/M1 pilot launchers remain unchanged for reproducibility.
- Production commands pin GDN chunk size 32 so generic precision-dependent CLI defaults cannot silently mutate the accepted checkpoint-visible model configuration.
- Regression contracts cover the 8,000/8,192 semantic-vs-physical vocabulary split, intrinsic Quantile selection, failed-attempt Quantile rollback semantics, production Top-K telemetry, fixed provider command identity, and the corpus-token bound (`7999` accepted; `8000` rejected).

The accepted SuperBPE 100B data path was subsequently wired on `moe-8e-top1` through head `c3863a0c4ba644320c6d28fe8f259e0243b5c87f`:

- `dataset/superbpe_retokenization.py` converts accepted ClimbMix records at the document boundary: optional terminal GPT-2 EOD removal -> exact GPT-2 byte decode -> strict UTF-8 reconstruction -> frozen SuperBPE encode. The converted token tuple is what the existing `SourceDocument` and deficit scheduler see.
- Ordinary pretraining source text can emit only SuperBPE BPE IDs `0..7991`. Reserved/control strings such as literal `<think>` are not interpreted as control tokens during corpus encoding; EOD `7992` is inserted only by the existing packer.
- The producer fails closed if the frozen tokenizer artifact no longer matches Git blob `a4daaa638a4d9db10270a65e8a6b55a9b94a9fd4`, validates the exact special-token inventory, records a runtime SHA-256, and includes tokenizer/vocab/EOD identity in SuperBPE configuration/schema/resume hashes.
- Legacy GPT-2 production hash inputs are preserved exactly so this new feature does not invalidate historical GPT-2 producer resume state.
- The final manifest records the tokenizer contract, `semantic_vocab_size=8000`, `eod_token_id=7992`, and SuperBPE as the source-token accounting unit. Full shard verification uses the manifest semantic-vocabulary bound rather than the historical GPT-2 size.
- `dataset/moe_100b.py` freezes a dedicated `moe-100b-superbpe-b64-dataset-001` profile at 100B SuperBPE source tokens, 90B/110B min/max, context 2048, 64 sequences/block, 1 GiB shards, 500M-source-token durable checkpoints, incremental READY publication and existing HF Storage Bucket durability.
- `modal/moe_100b_dataset.py` is a CPU-only resumable producer entrypoint using the same production pipeline and HF upload/frontier machinery; it does not launch the MoE training job.
- Targeted regression tests were added for retokenization-before-accounting, literal reserved-token text, EOD replacement, tokenizer-dependent resume hashes, semantic-vocabulary verification and the frozen 100B profile.

No GitHub Actions workflow/status checks are attached to branch head `c3863a0c4ba644320c6d28fe8f259e0243b5c87f`. A local clean-checkout test attempt from the assistant sandbox could not run because that environment could not resolve `github.com`; therefore the new tests are **written but not yet executed in a qualified project environment**. No 100B corpus production has been launched yet. The exact 64E/Top-2 GPU qualification also remains pending, so the long MoE training launch is still gated.
