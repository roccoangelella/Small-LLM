# S0 Training Architecture Decision

_Last updated: 2026-08-06 Europe/Rome_

## Scope

This decision record supersedes conflicting candidate defaults in `sft_design_freeze.md` for the first S0 run on the approximately-20M model pretrained on the approximately-100M-token finite dataset. It preserves unresolved questions explicitly rather than silently promoting recommendations.

## Frozen model and update scope

- Load the frozen native base checkpoint without changing model geometry.
- Keep the GDN-2/MHA schedule, dimensions, context length, tokenizer, tied embedding/head, normalization, recurrence, attention implementation, and FP16 execution unchanged.
- Use full-parameter fine-tuning. No LoRA, adapters, frozen layers, or new prediction head in S0.
- Initialize a new optimizer, scheduler, and GradScaler. Never restore pretraining optimizer moments or scheduler/scaler state.

## Frozen optimizer baseline

Use the same hybrid whole-matrix Muon + AdamW optimizer architecture and fail-closed parameter routing used in pretraining:

- ordinary feature-transform matrices remain on Muon;
- embeddings, norms, biases, GDN dynamics, and structured temporal filters remain on AdamW;
- Muon momentum and AdamW moments start from zero;
- retain the qualified Muon mechanics and routing identity;
- use zero weight decay on both branches for S0;
- retain global gradient clipping and fail-closed FP16 overflow behavior.

The initial S0 peak/base learning rate is `3e-5`. It is a first authorized value, not a claim of optimality; the implementation must keep it configurable.

## Frozen optimizer-update geometry

For scientific continuity, retain the pretraining effective target-token update size:

```text
32,768 loss-bearing target tokens per optimizer update
```

For chat examples, only active supervised targets count toward this number. For replay examples, every valid next-token target counts. Conversation counts and serialized context tokens do not define the optimizer block.

One optimizer block may contain variable-length records and be processed through multiple GPU microbatches. The trainer must sum losses over the whole block, divide once by the exact active-target count, and then perform one optimizer update. Checkpoints are legal only after the atomic block commits.

At a 4,000,000-target S0 horizon this yields 122 full-size-equivalent updates plus a final partial update when complete records leave a remainder.

## Scheduler direction

The user asked to reuse the same token-count WSD scheduler policy used in pretraining. This is technically compatible and is the recommended pending freeze:

- fresh scheduler state;
- warmup `max(16 updates, 5% of planned active targets)`;
- stable phase;
- cosine decay over the final 20% of planned active targets;
- minimum LR ratio `0.1`;
- advance only on successfully committed loss-bearing targets.

For a 4M target horizon and 32,768-target update geometry, the candidate exact horizons are:

```text
peak LR:       3e-5
minimum LR:    3e-6
warmup:        524,288 active targets (16 full updates)
decay:         800,000 active targets
stable:        2,675,712 active targets
```

The scheduler policy remains marked pending explicit confirmation because the minimum-16-update rule makes warmup approximately 13.1% of this short SFT run.

## Frozen S0 data/objective controls

- Total horizon: 4,000,000 loss-bearing targets for the 20M/100M experiment.
- One finite pass; implicit repetition is forbidden.
- Data budget, source ratios, and checkpoint horizons must be configuration values and scale to later, larger models.
- Overall target mixture: 85% filtered Smol-SmolTalk instruction targets and 15% frozen ClimbMix replay targets.
- Within the instruction portion, the current source target is 75% `smol-magpie-ultra-short`, 10% `smol-contraints`, 7.5% `smollm-rewrite-30k`, and 7.5% `smol-summarize-20k`, measured by assistant loss-bearing targets.
- The 85/15 and internal source shares must be easy to change without modifying trainer code.
- Maximum assistant target length: 512 tokens. Do not truncate an assistant target to fit; shorten old complete turns first, then reject if necessary.
- The source order is a deterministic seeded random interleaving that preserves the assigned target-token distribution over the stream and, as closely as practical, inside evaluation intervals and optimizer blocks.
- Use the same primary seed and repeatability logic as pretraining. S0 qualification does not require a multi-seed scientific claim.

## Frozen loss accounting

The trainer must support an explicit per-token supervision mask.

Baseline chat accounting under consideration remains:

- system, user, and role-marker targets masked;
- assistant content and turn-termination targets supervised;
- all valid replay next-token targets supervised.

For every atomic optimizer block, accumulate the sum of token losses over active targets and divide once by the block's exact active-target count. Never average per-example or per-microbatch means when active-target counts differ.

Prompt-token weighting and Dynamic Fine-Tuning are not yet frozen. The conventional response-only cross-entropy objective remains the recommended baseline pending the research decision.

## Frozen ordering, lifecycle, and checkpoints

- Reuse the pretraining fail-closed lifecycle: atomic updates, token-count scheduling, overflow retries without cursor advancement, exact checkpoint/resume, immutable data identity, and remote recovery.
- Use one finite pass and retain intermediate checkpoints rather than assuming the final checkpoint is best.
- Mandatory model-selection horizons remain 0.5M, 1M, 2M, and 4M committed loss-bearing targets.
- `Early checkpoint selection` means the run may complete through 4M while the selected S0 model may be an earlier checkpoint if instruction gains saturate or base retention/generation quality worsens. Hard failures may still stop training immediately.
- Exact local checkpoint, validation, and remote-publication step cadences are not copied numerically from pretraining because the complete S0 run has only about 123 updates; they remain to be frozen around the target-token milestones and measured overhead.
- SFT checkpoints must preserve the complete existing state plus parent base-checkpoint identity, SFT dataset/manifest identity, chat-template identity, supervision-mask schema, source-mixture state, committed active-target count, and exact next record/block.

## Implementation requirements kept open but mandatory to resolve

### Prompt-token loss

Public SFT practice commonly masks prompt tokens, while recent Weighted Instruction Tuning work reports gains from low-to-moderate prompt weights. S0 must implement a configurable prompt weight, but the production value is not frozen.

### SFT objective

Implement ordinary masked cross-entropy as the reference objective. Dynamic Fine-Tuning may be implemented behind an objective switch, but whether it receives a production comparison is unresolved. DFT downweights low-probability reference tokens using a detached student probability and has strongest published evidence on 1.5B-8B reasoning/code/multimodal settings, not a 20M general-chat model. Its interaction with a mixed standard-CE replay branch also requires an explicit normalization decision.

### Chat serialization

Kimi K3 and DeepSeek-V4 use reserved structural tokens, explicit role boundaries, assistant-turn stop markers, and separate reasoning/response/tool channels. Those tokens were available to the frontier models during their original training. The current Small LLM tokenizer did not reserve a chat protocol before pretraining.

For S0, the recommended pending decision is therefore a versioned plain-text role template using existing GPT-2 tokens and `<|endoftext|>` as the assistant-turn terminator, rather than introducing randomly initialized semantic tokens after pretraining. The exact byte-level template and whether every assistant turn or only the final turn receives `<|endoftext|>` remain to be frozen.

The SFT serialization interface must be pluggable so future from-scratch larger models can reserve dedicated role, turn, reasoning, and tool tokens before pretraining.

### Padding, bucketing, and packing

Recommended pending contract:

- one conversation per independent sequence;
- dynamic right padding to the longest sequence in each microbatch;
- local length bucketing inside a bounded shuffle/optimizer-block buffer to reduce padding waste without changing source quotas;
- no cross-conversation packing until both causal-attention isolation and explicit GDN recurrent-state reset at each packed boundary are implemented and proven.

### Post-SFT weight interpolation

Implement a post-training utility that can evaluate `theta_alpha = theta_base + alpha * (theta_sft - theta_base)` for selected alpha values. This scales back the total SFT displacement without retraining and may recover base retention, but no interpolation coefficient or automatic use is frozen.

### Model-selection scorecard

The exact combined chat-quality/base-retention score and acceptance thresholds are explicitly deferred to a dedicated discussion. Training loss alone may not select the model.

## Research references for open decisions

- Wu et al., `On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification` (ICLR 2026): Dynamic Fine-Tuning.
- Chatterjee et al., `On the Effect of Instruction Tuning Loss on Generalization` (TACL 2025): Weighted Instruction Tuning.
- Kimi Team, `Kimi K3: Open Frontier Intelligence` (2026): XTML chat protocol and post-training pipeline.
- DeepSeek-AI, `DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence` (2026): role/reasoning/tool encoding and expert-to-unified post-training.
