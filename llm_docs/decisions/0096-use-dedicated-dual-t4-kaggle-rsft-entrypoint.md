---
status: accepted
date: 2026-08-18
---

# ADR 0096 — Use a dedicated dual-T4 Kaggle entry point for R-SFT

## Context

The first R-SFT experiments inherit most of the operational requirements already qualified for S0: exact global-token optimizer steps, Kaggle's two Tesla T4 GPUs, FP16 prewarm, WSD training, checkpoint cadence, exact resume, W&B telemetry, and Hugging Face publication. R-SFT differs in its parent checkpoint, tokenizer/model transition, bundle budget semantics, and delimiter experiment.

The first R0 corpus is intentionally bundle-driven and one-pass. Its final production token budget is not frozen as a percentage of pretraining. The initial delimiter experiment compares textual and atomic reasoning boundaries under matched training conditions.

## Decision

- Add a dedicated human entry point at `kaggle/launch_r_sft.py` for R-SFT training.
- The first supported profile is the 100M model whose pretraining parent consumed approximately 2B tokens, and whose direct parent is completed S0 run `100m-2b-sft-s0-001`.
- Execute R-SFT on Kaggle with exactly two Tesla T4 GPUs by reusing the qualified `dual_t4_sft.py` DDP/trainer machinery rather than forking a second training engine.
- R-SFT consumes the immutable tokenized bundle exactly once. `train_target_tokens_requested` from the verified bundle is the training target budget; do not reinterpret it as the S0 4% parent-fraction rule.
- Before training, promote the S0 model to semantic vocabulary 50,260 using the existing R-SFT model-transition contract. Both textual and atomic delimiter arms use the same promoted architecture and initialization.
- Require an explicit reasoning-token spec file. The three promoted token IDs remain fixed, but their strings remain externally configured until the delimiter/token spelling decision is frozen.
- Carry the reasoning-tokenizer metadata and R-SFT delimiter identity in checkpoint pipeline state so R-SFT chat/inference can reconstruct the tokenizer contract.
- Give textual and atomic arms distinct checkpoint template identities while keeping their training machinery otherwise matched.
- Do not treat the existing S0 inline behavior probe as an R-SFT qualification suite. Mark that probe skipped for these runs; the dedicated R-SFT qualification suite will be framed separately.
- Preserve the existing SFT operational behaviors for DDP synchronization, checkpointing, verified resume, W&B, and Hugging Face remote publication.

## Consequences

The public Kaggle command is R-SFT-specific while the expensive/fragile GPU execution path remains shared with the already-qualified SFT implementation. The launcher fails closed on missing token specs, bundles, parent repositories, or unsupported model profiles. R-SFT run IDs remain explicit so the textual/atomic pilot and later production run cannot collide in checkpoint or W&B namespaces.
