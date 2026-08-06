# S0 Training Architecture Decisions

_Last updated: 2026-08-06 Europe/Rome_

## Scope

This record freezes the user's decisions for the first S0 supervised fine-tuning experiment on the approximately-20M-parameter checkpoint pretrained on approximately 100M tokens. It also records the remaining open questions and the current research-backed recommendations without silently treating recommendations as approved decisions.

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

## Frozen loss accounting

- Ordinary baseline objective remains token-level cross-entropy unless the user separately approves Dynamic Fine-Tuning.
- Average the loss only over active loss-bearing targets.
- User, system, role-label, padding, and other masked positions do not enter the denominator.
- Chat assistant targets have weight `1.0`.
- ClimbMix replay targets have weight `1.0`.
- Prompt-token loss weight is provisionally `0.0`; the final prompt-loss decision remains research-informed but not separately frozen by this record.

## Frozen data mixture and ordering

- Keep the 85% instruction / 15% ClimbMix replay target-token mixture.
- The implementation must expose these shares as configuration values for future experiments and larger models.
- Within instruction data, use the previously proposed source-level target-token shares unless superseded by the pinned-data audit:
  - 75% `smol-magpie-ultra-short`
  - 10% `smol-contraints`
  - 7.5% `smollm-rewrite-30k`
  - 7.5% `smol-summarize-20k`
- Randomize records deterministically without replacement while preserving the target-token source mixture across atomic blocks and evaluation horizons.
- The schedule is source-stratified by target-token deficit, not semantically classified by a new TF-IDF capability model.

## Frozen length and sequence policy

- Maximum assistant target length: 512 tokens.
- Context remains 2,048 input tokens.
- Use one conversation per sequence for S0 v1.
- Do not cross-pack unrelated conversations until both causal-attention isolation and explicit GDN recurrent-state resets are implemented and parity-tested.
- Use right padding only; padding tokens and their targets are masked.
- Dynamic padding and deterministic length bucketing may reduce wasted compute without changing source membership, atomic-block composition, or target-token accounting.

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

## Open: prompt-token loss

The ordinary S0 recommendation is assistant-only loss plus full next-token loss on replay. Recent Weighted Instruction Tuning research reports that a small prompt-token weight can improve robustness in tested larger-model settings, but this is not yet treated as the dominant default and has not been validated on a lightly pretrained 20M model. The implementation should support a configurable prompt-loss weight, while the baseline remains `0.0` unless the user changes it.

## Open: standard cross-entropy versus Dynamic Fine-Tuning

Dynamic Fine-Tuning multiplies each target token's cross-entropy by the model's detached probability for that target. This suppresses gradients from very low-probability targets and emphasizes medium-confidence targets. Published gains focus primarily on reasoning, code, and other high-entropy trajectories, while the authors explicitly report weaker or inconsistent behavior in some low-entropy and non-reasoning domains. Later work adds compatibility control or anchoring because plain DFT can be sensitive to demonstration-policy mismatch and drift.

Current recommendation, pending user approval:

```text
S0 baseline: ordinary masked token cross-entropy
optional controlled ablation: DFT on the identical base checkpoint, data order, target budget, optimizer, and schedule
```

## Open: exact chat serialization

Recent frontier templates use explicit role boundaries, an explicit end marker for each assistant turn, and separate reasoning/response channels where reasoning is supported. Kimi K3 uses tagged messages plus an end-of-message token; DeepSeek-V4 uses BOS, user and assistant control tokens, EOS after assistant turns, and optional `<think>...</think>` structure.

Because S0 currently avoids vocabulary expansion, the recommended Small LLM equivalent is pending approval:

```text
<|endoftext|>System:\n{system}\n\nUser:\n{user}\n\nAssistant:\n{assistant}<|endoftext|>
User:\n{next_user}\n\nAssistant:\n{next_assistant}<|endoftext|>
```

Recommended rules:

- omit the system section when absent;
- use the exact `Assistant:\n` generation prefix;
- supervise assistant content and the following end-of-text token for every assistant turn;
- mask system, user, and role-label text;
- use no reasoning tags in S0;
- preserve byte-exact template tests;
- treat the end-of-text token as the generation stop marker.

This per-assistant-turn stop policy supersedes the earlier provisional idea of supervising only one final EOS if the user approves it.

## Open: model-selection scorecard

The user explicitly deferred the exact model-selection metric and retention thresholds to a dedicated future discussion. The implementation must still emit the necessary instruction, generation, intrinsic, retention, and operational metrics so that the later decision does not require retraining.

## Optional post-training utility: base/SFT interpolation

Weight interpolation is not yet frozen as the selected output model. It is a cheap evaluation utility:

```text
theta(alpha) = theta_base + alpha * (theta_sft - theta_base)
```

`alpha=1` is the full SFT checkpoint; lower values scale back the complete SFT update direction toward the base model. Evaluating a small grid can reveal whether some chat behavior is retained while reducing base-model regression. It requires no additional training and must never replace explicit checkpoint evaluation.

## Early checkpoint selection

Early checkpoint selection means the final selected S0 model need not be the 4M-target endpoint. The run may complete the authorized one-pass horizon while preserving checkpoints at earlier target counts. If chat gains saturate or base retention worsens after an earlier checkpoint, that earlier checkpoint can be selected. This is model selection from a completed trajectory, not necessarily process-level early termination.
