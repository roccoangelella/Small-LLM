# S0 Training Architecture Decisions

_Last updated: 2026-08-06 Europe/Rome_

## Scope

This record freezes the user's decisions for the first S0 supervised fine-tuning experiment on the approximately-20M model pretrained on approximately 100M tokens. The model architecture remains unchanged. S0 is a full-parameter continuation from the frozen base weights with a fresh optimizer and SFT-specific data/loss semantics.

## Frozen decisions

1. Use full-parameter fine-tuning. No LoRA, adapters, or frozen layers in S0.
2. Reuse the exact hybrid whole-matrix Muon + AdamW optimizer family, parameter routing, momentum, target-direction-RMS policy, clipping policy, and optimizer safety checks used in pretraining. Initialize all optimizer state from zero; do not restore pretraining moments.
3. Use a peak learning rate of `3e-5` for the first S0 baseline. This may be revised only by a later explicit decision based on pilot evidence.
4. Use zero weight decay for S0.
5. Reuse the pretraining scheduler policy in token-count form: warmup, stable phase, then cosine decay to one tenth of peak LR. Recompute all horizons from the finite 4M loss-bearing-target-token S0 plan; never resume the pretraining scheduler state.
6. Keep the same effective optimizer target-token block used in pretraining for scientific control, interpreted for SFT as the same number of loss-bearing target tokens per atomic optimizer update. The implementation must sum loss over active targets across all microbatches and divide once by the exact active-target count.
7. Standard S0 loss is ordinary causal cross-entropy normalized only over loss-bearing targets.
8. Baseline prompt-token loss weight is `0.0`; assistant response and final EOS weight are `1.0`; replay next-token targets are `1.0`. Prompt-loss weighting remains an implemented future ablation, not the baseline.
9. Ordinary cross-entropy is the S0 scientific baseline. Dynamic Fine-Tuning, Compatibility-Aware DFT, anchored DFT, token-adaptive reweighting, and related objectives remain optional later comparisons. The loss interface should make alternate token weights possible without changing dataset identity.
10. Keep the 85% instruction / 15% ClimbMix replay target-token mixture and make both shares configuration values rather than code constants.
11. Use one frozen, byte-exact S0 chat template with the existing GPT-2 vocabulary and no new special tokens. Separate hidden reasoning fields, preserved-thinking history, tool-call syntax, multimodal markers, and reasoning-effort modes used by frontier agentic models are out of scope for S0.
12. Maximum assistant target length is 512 tokens. Never truncate an assistant target; shorten only by removing oldest complete context turns, otherwise reject.
13. Use one conversation per sequence, right padding, dynamic padding to the longest record in each microbatch, and length-bucketed batching. Cross-conversation packing is disabled until both attention isolation and GDN recurrent-state reset are implemented and parity-tested.
14. Use deterministic pseudorandom ordering while preserving the frozen source/replay token distribution throughout the run. Randomness must be seed-controlled and resumable.
15. Use one finite pass and the same fail-closed stopping, checkpoint-boundary, durability, resume, and publication principles used in pretraining. Evaluation/checkpoint milestones are expressed in committed loss-bearing target tokens.
16. Implement post-SFT base/SFT checkpoint interpolation as an evaluation tool, not an automatic deployment step. Candidate interpolation weights must be evaluated against both chat behavior and base retention.
17. Use early-checkpoint selection: evaluate intermediate checkpoints and permit selecting 0.5M, 1M, or 2M rather than the final 4M checkpoint when later training improves SFT loss but worsens general retention or generation behavior.
18. Apply the same repeatability logic as pretraining: deterministic identities, a fixed primary seed, exact resume tests, and later confirmation with an additional seed before scientific claims.
19. Final model-selection metric and numeric retention gates are explicitly deferred to a dedicated future discussion.
20. SFT checkpoints must include full model, optimizer, scheduler, GradScaler, RNG, data cursor, committed loss-bearing-token counter, parent base checkpoint, dataset manifest, template identity, loss schema, mixture policy, and code identity.

## Scheduler interpretation

Reusing the pretraining scheduler means reusing its policy, not its state or absolute token horizons. The S0 schedule starts from zero with peak LR `3e-5`, uses the same warmup/stable/cosine-decay shape, and derives integer boundaries from the exact S0 optimizer-block plan. The scheduler advances only after a successful atomic optimizer update and by committed loss-bearing target tokens.

## Chat-template boundary

Kimi K3 and DeepSeek V4 support frontier-specific reasoning modes and structured message histories. Kimi K3 retains separate `reasoning_content` and visible `content`, preserves reasoning history across turns, and uses model-specific control-token/XTML serialization. DeepSeek V4 exposes non-think, think-high, and think-max modes. These designs are not suitable templates for S0 because the project has a frozen GPT-2 vocabulary, no tool or multimodal scope, and deliberately defers reasoning distillation. S0 therefore uses a minimal text-only System/User/Assistant serialization and final EOS, with assistant-only supervision.

## Open implementation parameters

The following are implementation details to derive from the existing pretraining contract or bounded hardware qualification, not unresolved conceptual architecture decisions:

- exact integer warmup/stable/decay target-token boundaries;
- exact checkpoint and validation cadence in target tokens while retaining pretraining durability policy;
- microbatch size and length buckets on the T4;
- source-level filter thresholds and final pinned dataset revision;
- numeric model-selection score and forgetting gates, deferred by decision 19.

## Research interpretation

Assistant-only masking remains the conventional and clean baseline. Published work on weighted instruction tuning reports potential benefits from small nonzero prompt-token weights, but results depend on task and completion length. Dynamic Fine-Tuning rescales token gradients using current policy probabilities and has shown improved generalization in larger reasoning-oriented settings. Subsequent work identifies drift, instability, and demonstration-policy compatibility limitations, motivating anchored and compatibility-aware variants. Therefore these objectives are implemented as future ablations rather than replacing ordinary cross-entropy in the first 20M engineering qualification.
