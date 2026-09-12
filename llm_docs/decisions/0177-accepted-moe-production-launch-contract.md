---
status: accepted
date: 2026-09-11
supersedes: null
---

# 0177 — Accepted MoE production launch contract

## Context and problem statement

No separate context was recorded at the time; section added for the template.

## Considered options

No alternative was recorded at the time; section added for the template.

## Decision outcome

1. The frozen SuperBPE tokenizer is the vocabulary source of truth for the production MoE model.
   - Exact semantic vocabulary: **8,000 token IDs** (`0..7999`).
   - Composition: **7,192 word-level byte-BPE entries + 800 SuperBPE entries + 8 special tokens = 8,000**.
   - `word_merges = 6,949` is the number of word-level merge operations, not the vocabulary size.
   - `semantic_vocab_size = 8000`.
   - `padded_vocab_size = 8192` exists only for physical tensor alignment/storage; IDs `8000..8191` are not semantic tokens and must not be valid output classes or corpus IDs.

2. Production training must explicitly use `MoEModelConfig.accepted()` rather than the historical `substantive()` 8-expert/Top-1 pilot configuration.

3. **Quantile Balancing is intrinsic to the accepted production model.** A generic CLI default must not silently disable or replace it. Experimental/smoke configurations may remain configurable separately.

4. The accepted production identity is **64 experts, Top-2, expert hidden width 352**, all transformer blocks MoE, sqrt-softplus routing, unbiased selected-score mixture weights, no shared expert, no auxiliary balancing loss, no router z-loss, router FP32/AdamW, dropless padded batched expert GEMM.

5. Dedicated provider launch paths must preserve that accepted model identity and must not reconstruct the old bounded M0/M1 pilot configuration. Production targets are Modal H100 and Beam RTX4090; historical pilot launchers remain unchanged for reproducibility.

6. Before any long production training run, the exact accepted path must pass a short qualification covering forward/backward/update, 64E/Top-2 routing, Quantile behavior, semantic-vocabulary bounds, checkpoint/resume, and production telemetry.

7. Quantile-Balancing controller semantics:
   - All gradient-accumulation microbatches belonging to one logical optimizer step use the same currently committed selection bias.
   - During those microbatches, the controller accumulates the per-expert raw-score frontier required for the exact `K/E` quantile.
   - After a successful optimizer step, expert threshold `q_e` is computed and the selection bias is replaced by `mean(q) - q_e`.
   - An expert with a higher natural threshold therefore receives a lower bias and becomes less likely to enter Top-K on the following logical step.
   - Bias affects Top-K selection only. Mixture weights are computed from the original unbiased selected scores.
   - If an optimizer attempt is skipped/retried because of overflow or another non-finite update, the transient score frontier is discarded and the previously committed bias remains unchanged. A failed update must have no persistent influence on later routing.

8. The full 100B pretraining corpus for this MoE line must be regenerated with the frozen SuperBPE tokenizer; GPT-2-tokenized training shards are not valid inputs. The exact retokenization/data-production contract is recorded separately in ADR 0176.

## Wiring status

Implemented on `moe-8e-top1` beginning with production-wiring head `d8bd2ca3880a10b05a133f75a32c2ad322c5fb47` and extended by later data-pipeline commits:

- `MoEModelConfig.accepted()` uses semantic vocab 8000 / padded vocab 8192, 64 experts, Top-2, expert width 352, sqrt-softplus routing, Quantile Balancing and batched dropless expert GEMM.
- `python -m MOE_model --model-size accepted` is an explicit production identity.
- The generic balancing CLI no longer defaults the accepted path to `none`; conflicting balancing overrides are rejected.
- Production telemetry reports configured Top-K rather than the historical hard-coded Top-1 value.
- Provider-neutral accepted execution plus dedicated Modal H100 and Beam RTX4090 launchers are present; pilot launchers remain separate.
- Production commands pin GDN chunk size 32 so precision-dependent generic defaults cannot silently mutate checkpoint-visible accepted geometry.
- Regression contracts cover the 8,000/8,192 semantic-vs-physical vocabulary split, intrinsic Quantile selection, failed-attempt rollback, Top-2 telemetry and accepted provider command identity.

## Qualification status

The accepted production wiring is implemented but the exact 64E/Top-2 GPU qualification has not yet been executed. No GitHub Actions workflow/status checks are attached to this branch. Therefore the long production training run remains gated on the short exact-configuration GPU qualification.

## Consequences

### Branch ownership

This contract belongs to the MoE development line and is canonical in `llm_docs/` on `moe-8e-top1`. It must not be duplicated into `main` unless/until the MoE line is intentionally merged there.
