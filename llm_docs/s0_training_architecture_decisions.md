# S0 Training Architecture Decisions

_Last updated: 2026-08-06 Europe/Rome_

## Scope

This record freezes the user's decisions for the first S0 supervised fine-tuning experiment on the approximately-20M-parameter checkpoint pretrained on approximately 100M tokens. Recommendations are not silently treated as decisions. Superseded proposals are retained where they explain the current design.

## Frozen model and update scope

- Use the exact pretrained model architecture and tokenizer unchanged.
- Perform full-parameter fine-tuning; do not use LoRA, adapters, or frozen layers for S0.
- Load the pretrained model weights only.
- Initialize fresh optimizer, scheduler, and GradScaler state for S0.
- Keep the same hybrid whole-matrix Muon + AdamW optimizer family and the same validated parameter-routing contract used in pretraining.
- Keep the pretraining Muon and AdamW mechanics unless a concrete S0 stability failure requires a separately recorded change.
- Use zero weight decay during S0.
- Begin with a peak base learning rate of `3e-5`.

## Frozen scheduler policy

Use the same token-count WSD policy as pretraining, but initialize a fresh schedule over the S0 target-token horizon:

```text
schedule: warmup / stable / cosine decay
warmup: max(16 successful optimizer updates, 5% of planned updates)
decay: final 20% of planned committed loss-bearing target tokens
minimum LR ratio: 0.1
scheduler state restored only when resuming S0, never from pretraining
```

The exact horizons must be derived from the verified S0 manifest. With a nominal 4M-target horizon and 32,768 active targets per full optimizer update, the run has approximately 122 full updates plus a possible final partial block, so the exact integer schedule must be frozen from the final block plan.

## Frozen optimizer-block scale

For controlled comparison with pretraining, keep the same effective optimizer target scale:

```text
approximately 32,768 loss-bearing target tokens per optimizer update
```

This is defined by active supervised targets, not by number of conversations or total serialized tokens. Chat examples contribute assistant-response and supervised end-of-turn targets; replay examples contribute ordinary next-token targets. Variable-length microbatches accumulate summed loss until the complete atomic block is processed, then divide once by the exact number of active targets and perform one optimizer update.

## What `assistant` means

`assistant` is a role in the logical conversation schema. It identifies text authored by the model side of the dialogue. It is not a separate network, head, loss module, or literal special token.

For S0, an example record conceptually contains:

```text
system: optional instructions governing the dialogue
user: input supplied to the model
assistant: the desired model response
```

The complete formatted sequence is processed so the model can condition on the system and user text. The baseline SFT loss is applied only to the assistant message content and the end-of-turn marker immediately following that content. The literal role label `Assistant:\n` is context and remains masked.

Example:

```text
<|endoftext|>User:
What is the capital of France?

Assistant:
Paris is the capital of France.<|endoftext|>
```

Loss policy:

```text
<|endoftext|>                         masked
User:\n                               masked
What is the capital of France?       masked
Assistant:\n                          masked
Paris is the capital of France.      loss-bearing
<|endoftext|>                         loss-bearing
```

In a multi-turn conversation, every assistant response and its following end marker are supervised. At inference, the prompt ends after `Assistant:\n`; the model then generates the assistant content until `<|endoftext|>`.

In S0, assistant responses come from the retained open dataset. In later stages, the same schema can hold teacher-generated or distilled responses without changing the model architecture.

## Frozen loss accounting

- Use ordinary masked token-level cross-entropy as the S0 baseline.
- Average the loss only over active loss-bearing targets.
- User, system, role-label, padding, and other masked positions do not enter the numerator or denominator.
- Assistant-content targets have weight `1.0`.
- The end-of-turn `<|endoftext|>` following every assistant response has weight `1.0`.
- ClimbMix replay targets have weight `1.0` and use the ordinary next-token objective.
- Prompt-token loss weight is frozen at `0.0` for S0.

The implementation must keep target weights configurable, but changing them creates a new objective and requires a separately recorded experiment.

## Deferred improvement: Dynamic Fine-Tuning

Dynamic Fine-Tuning, or DFT, reweights each supervised token's cross-entropy using the model's detached probability for that target. It suppresses gradients from targets the current model considers extremely improbable and has reported benefits primarily in reasoning, code, and heterogeneous teacher-trajectory settings.

DFT is not the S0 baseline. The user decided:

```text
S0: ordinary masked cross-entropy
larger future SFTs: retain DFT as an optional controlled improvement
```

The loss interface should therefore support future objective selection without changing dataset serialization, checkpoint lineage, or the ordinary cross-entropy implementation. A future DFT comparison must start from the same parent checkpoint and hold data order, optimizer, schedule, target budget, and evaluation fixed.

## Frozen data mixture and ordering

- Keep the 85% instruction / 15% ClimbMix replay target-token mixture.
- Expose these shares as configuration values; do not hard-code them for future larger models.
- Within instruction data, use the previously proposed source-level target-token shares unless superseded by the pinned-data audit:
  - 75% `smol-magpie-ultra-short`
  - 10% `smol-contraints`
  - 7.5% `smollm-rewrite-30k`
  - 7.5% `smol-summarize-20k`
- Randomize records deterministically without replacement while preserving the target-token source mixture across atomic blocks and evaluation horizons.
- The schedule is source-stratified by target-token deficit, not semantically classified by a new TF-IDF capability model.

Randomness must not allow long source streaks that materially change the training distribution over time. The builder should independently shuffle each source with the frozen seed and choose records while tracking each source's deficit relative to its target-token share.

## Frozen chat serialization

S0 does not add vocabulary tokens. It reuses GPT-2 token `50256`, rendered as `<|endoftext|>`, as the beginning boundary and the assistant end-of-turn/stop marker.

With a system message:

```text
<|endoftext|>System:
{system message}

User:
{user message}

Assistant:
{assistant response}<|endoftext|>
```

Without a system message:

```text
<|endoftext|>User:
{user message}

Assistant:
{assistant response}<|endoftext|>
```

A later user turn is appended directly after the previous assistant end marker:

```text
User:
{next user message}

Assistant:
{next assistant response}<|endoftext|>
```

Frozen rules:

- omit the entire system section when absent;
- use the exact `Assistant:\n` inference prefix;
- supervise assistant content and the following `<|endoftext|>` for every assistant turn;
- mask system, user, and role-label text;
- use no reasoning tags in S0;
- treat `<|endoftext|>` as the generation stop marker;
- require byte-exact golden serialization and mask tests;
- training, validation, and inference must use the same serializer implementation.

This per-assistant-turn stop policy supersedes the earlier provisional idea of supervising only one final EOS for the complete conversation.

## Frozen length and sequence policy

- Maximum assistant target length: 512 tokens, excluding the following end-of-turn marker.
- Context remains 2,048 input tokens.
- Use one conversation per sequence for S0 v1.
- Do not cross-pack unrelated conversations until both causal-attention isolation and explicit GDN recurrent-state resets are implemented and parity-tested.
- Use right padding only; padding tokens and their targets are masked.
- Use dynamic padding to the longest real sequence in each GPU microbatch.
- Use deterministic length bucketing to reduce padding waste without changing source membership, atomic-block composition, target-token accounting, or the global randomized order.

Length bucketing is an execution optimization, not a curriculum. Records are first assigned to the deterministic atomic optimizer block; only their microbatch arrangement inside that block may be reordered by length.

## Frozen pass, stopping, checkpointing, and repeatability logic

- Train for one finite pass over the approved S0 stream.
- Implicit repetition or wraparound is forbidden.
- Reuse the pretraining fail-closed checkpoint and resume principles:
  - commit only completed optimizer blocks;
  - count only successfully committed active target tokens;
  - restore the exact next block and record;
  - preserve model, optimizer, scheduler, GradScaler, RNG, dataset cursor, and identity state;
  - publish latest and validation-selected checkpoint identities;
  - require a final verified checkpoint.
- Exact numerical checkpoint and validation cadence must be scaled to the much shorter S0 run rather than blindly copying the 100M-pretraining step cadence.
- Use the same deterministic-seed and repeatability philosophy as pretraining, including the existing seed unless a separate decision changes it.

## Frozen checkpoint contents

Every S0 checkpoint must identify and restore:

```text
model parameters
Muon state
AdamW state
scheduler state
GradScaler state
RNG states
successful optimizer-block count
committed loss-bearing target count
dataset block and record cursor
source-mixture schedule state
parent base-checkpoint identity
S0 dataset manifest identity
chat-template identity
loss-objective identity
filter and source configuration
git commit and model geometry
```

## Configurable post-training utility: base/SFT interpolation

The default evaluated checkpoint uses the complete SFT weights:

```text
alpha: 1.0
```

The SFT module must make interpolation easy to configure:

```text
theta(alpha) = theta_base + alpha * (theta_sft - theta_base)
```

`alpha=1.0` is the unmodified SFT checkpoint. Lower values move the checkpoint back toward the parent base model and may recover some base capability when full SFT causes measurable regression. Interpolation requires no additional training but every candidate must pass the same full evaluation and identity checks.

Interpolation is an optional evaluation/export utility, not a hidden modification of the training checkpoint. The parent base checkpoint identity and the chosen alpha must be recorded in every interpolated artifact.

## Early checkpoint selection

The last 4M-target checkpoint is not automatically the selected S0 model. Preserve checkpoints at the approved intermediate target horizons. If instruction gains saturate or base retention worsens after an earlier checkpoint, that earlier checkpoint may be selected even though the authorized one-pass run completed.

This is model selection from a completed trajectory, not necessarily process-level early termination.

## Deferred: model-selection scorecard

The user explicitly deferred the exact model-selection metric and retention thresholds to a dedicated future discussion. The implementation must still emit all required instruction, generation, intrinsic, retention, and operational metrics so that the later decision does not require retraining.
