# SFT Design Freeze

_Last updated: 2026-08-06 Europe/Rome_

## Decision status

The user decided to begin a deliberate design-freeze process for supervised fine-tuning and later post-training. This document is the working source of truth for that process.

Only the process decision is frozen by this initial record. Technical choices below are candidate defaults pending explicit user approval. They are deliberately separated into **proposed**, **deferred**, and **rejected for the first run** so recommendations are not silently recorded as accepted decisions.

## Purpose of the first SFT run

**Proposed default:** treat the approximately-20M-parameter checkpoint as an end-to-end post-training qualification model.

The first SFT run should test whether the project can reproducibly transform a frozen base checkpoint into a conversational checkpoint while preserving measurable base-language ability. It is not expected to establish the final reasoning ceiling, final data mixture, or final post-training objective for a larger model.

The first SFT run should answer:

1. Does the chat serialization and assistant-only loss contract work?
2. Can the model learn basic instruction following and response completion?
3. Does joint training with pretraining replay limit forgetting?
4. Can training, validation, checkpointing, interruption, resume, publication, and paired base-versus-SFT evaluation remain exact and auditable?
5. Does teacher-response distillation improve over an open-data SFT baseline when prompt identities and token budgets are held fixed?

## Recommended stage order

### Stage S0 — Open-data behavior SFT baseline

Use a small-model-oriented, openly licensed instruction/chat dataset plus a fixed replay slice from the project's pretraining distribution.

The preferred initial source is a filtered and pinned subset of `HuggingFaceTB/smol-smoltalk`, because it was explicitly constructed for the 135M- and 360M-parameter SmolLM2 models, excludes advanced examples that were inappropriate for those models, and is Apache-2.0 licensed. The project must still pin an exact dataset revision, retain source identifiers, and audit every retained subset.

Do not ingest the entire SmolTalk or Tulu mixtures. They include code, advanced mathematics, tool use, long-context material, mixed third-party terms, and task distributions that do not match the project's current English-chat scope.

### Stage S1 — Teacher-response distillation comparison

Use the same frozen prompt pool as S0, but replace selected assistant responses with concise outputs from an approved teacher. Keep the prompt split, token budget, replay ratio, template, optimizer-selection protocol, and evaluation fixed.

This makes distillation a controlled data-label comparison rather than a simultaneous change to prompts, task mixture, training budget, and objective.

Teacher-generated ordinary answers are already a form of offline response distillation. A separate knowledge-distillation loss is not required for this stage.

### Stage S2 — Capacity-aligned concise reasoning distillation

Only after S0 and S1 pass, add a small set of verifiable tasks with short explanations. Generate multiple correct candidate traces, reject incorrect traces, and prefer the shortest correct response that has relatively low negative log-likelihood under the frozen student base checkpoint.

The first reasoning curriculum should teach concise justifications, not long monologue-style chain of thought.

### Stage S3 — On-policy distillation or reinforcement learning

Deferred until the student already produces nonzero success and reward variance on a frozen set of verifiable tasks. Cold-start SFT must precede this stage.

## Why long chain of thought is deferred

A 20M model trained on approximately 100M source tokens is primarily an engineering qualification model. Long teacher traces can exceed the student's representational and optimization capacity, encourage imitation of verbose surface form, consume most of the limited supervised-token budget, and obscure whether failures come from the chat pipeline or reasoning transfer.

Recent work supports capacity-aligned distillation rather than indiscriminately copying long traces:

- Complexity-aware fine-tuning reports that reasoning supervision should be reserved for difficult samples rather than applied uniformly.
- SmartAD selects the most student-compatible correct teacher trajectory and emphasizes action/final-decision spans over intermediate reasoning.
- ReasonLite-0.6B uses a short-CoT stage before a long-CoT stage, but begins from a much stronger 0.6B base model and millions of curated verified traces.
- On-policy distillation studies report that cold-start alignment and compatible teacher/student thinking distributions are prerequisites for reliable transfer.

Therefore, long-CoT distillation is rejected for S0 and S1 and remains a later controlled experiment.

## Teacher policy

### ChatGPT and OpenAI output

**Proposed rejection:** do not use ChatGPT or the OpenAI API to generate training labels for this general-purpose model.

OpenAI's current consumer and business terms prohibit using Output to develop models that compete with OpenAI, and the consumer terms also prohibit automated or programmatic extraction of Output. Although users generally own their Output, ownership does not override these use restrictions. The project should not build a dataset whose permitted use is ambiguous or incompatible with its goal.

### Approved teacher classes

A teacher is eligible only when all of the following are recorded:

- exact model and version;
- exact provider or local-weight revision;
- applicable model, API, and output terms;
- prompt template and generation parameters;
- raw response provenance;
- filtering, judging, and verification results;
- permission to use outputs for training and model distillation.

Candidate teacher routes include:

1. An open-weight Apache-2.0 model such as a pinned Qwen3.5 checkpoint, subject to the terms of the actual inference provider if not run locally.
2. DeepSeek's API, whose current terms explicitly permit using outputs to train other models, including model distillation.

No teacher is frozen yet. Provider terms must be rechecked immediately before dataset generation.

## Proposed S0 data mixture

Measure mixture shares by **loss-bearing target tokens**, not raw examples.

```text
45% simple single-turn helpful answers and factual transformations
15% rewriting, summarization, extraction, and formatting
10% multi-turn everyday conversation
10% explicit constraint and instruction following
 5% concise verifiable explanation or elementary reasoning
15% original-distribution pretraining replay
```

Additional constraints:

- English only for S0.
- No deliberate coding capability.
- No tool calls or function-call syntax.
- No long-context tasks beyond the frozen 2,048-token model context.
- No advanced contest mathematics.
- No long chain-of-thought traces.
- Basic uncertainty, correction, and safe refusal examples may be included inside the ordinary instruction categories, but should not dominate the tiny model's behavior.
- At most 10% of instruction conversations should use a nonempty system message in S0.

The 15% replay share is a candidate baseline. The clean ablation set is 0%, 15%, and 30% replay, while 15% remains the proposed default. Replay data uses the original next-token objective and is not converted into fake chat conversations.

## Proposed token budget and split

Use token budgets rather than a fixed example count.

```text
total train loss-bearing target tokens: 4,000,000
instruction/chat target tokens:          3,400,000
pretraining replay target tokens:          600,000
training passes:                                  1
implicit repetition:                      forbidden
context length:                               2,048
```

Split prompt or document identities before teacher generation and before tokenization:

```text
train:      95.0%
validation:  2.5%
test:        2.5%
```

All derivatives of the same source prompt, conversation, document, or synthetic seed must remain in one split.

The exact 4M budget is proposed, not frozen. A tiny deterministic fixture and a bounded pilot must precede production training.

## Proposed conversation schema

Canonical logical schema:

```text
conversation_id
source_id
source_revision
license_id
messages[]:
  role: system | user | assistant
  content: string
assistant_span_metadata[]
teacher_metadata or null
quality_metadata
split
content_hash
```

Every record must preserve provenance. Missing or unrecognized licenses fail dataset construction.

## Proposed chat serialization

Do not add vocabulary items or randomly initialized role embeddings for the first SFT experiment. Reuse the existing GPT-2 tokenizer and its existing end-of-text token.

Canonical text form:

```text
System:
{system text}

User:
{user text}

Assistant:
{assistant text}

User:
{next user text}

Assistant:
{next assistant text}<|endoftext|>
```

Rules:

- Omit the entire `System:` section when no system message exists.
- End inference prompts immediately after `Assistant:\n`.
- Apply supervised loss to assistant content spans and the final end-of-text token only.
- Mask role labels, system text, and user text from the SFT loss.
- Supervise every assistant turn in a multi-turn conversation.
- Use only one end-of-text token at the end of the complete conversation.
- Preserve the system message and final user/assistant exchange when shortening an overlength conversation.
- Remove oldest complete user/assistant turn pairs first.
- If the final target response still does not fit without truncation, reject the sample.
- Never train on a partially truncated target response.

## Proposed packing and batching contract

SFT v1 should use one conversation per model sequence. Cross-conversation packing is rejected for the initial implementation because the hybrid model would require explicit attention and recurrent-state resets at every packed boundary to avoid information leakage.

Use padding masks and supervised-token masks. Accumulate gradients by supervised target-token count rather than by a fixed number of examples.

Proposed effective update target:

```text
approximately 16,384 loss-bearing target tokens per optimizer update
```

Replay examples count all valid next-token targets. Instruction examples count only unmasked assistant and final end-of-text targets.

## Proposed optimization contract

- Full-parameter SFT; no LoRA for the approximately-20M model.
- Start from a frozen native base checkpoint.
- Initialize a fresh optimizer; never resume pretraining optimizer moments.
- Use AdamW for the first SFT baseline rather than introducing a second Muon experiment into the behavior qualification.
- Candidate AdamW defaults: beta1 0.9, beta2 0.95, epsilon 1e-8, weight decay 0.0, global gradient clipping 1.0.
- FP16 with the existing fail-closed GradScaler and non-finite checks.
- One finite pass over the frozen dataset.
- Warm up for 3% of planned loss-bearing target tokens, then cosine decay to 10% of peak learning rate.

Do not freeze a peak learning rate before a bounded probe. Freeze the selection protocol instead:

```text
candidate peak LRs: 3e-5, 1e-4, 3e-4
probe budget:       first 5% of the frozen S0 training tokens
selection:          highest stable LR that passes both instruction-learning
                    and base-retention gates
```

The probe uses identical data order and initialization for all candidates. The selected LR is recorded before the complete run.

## Proposed evaluation and gates

Every SFT checkpoint report must pair post-training metrics with the immutable base-checkpoint metrics.

### Instruction behavior

- assistant-prefix completion rate;
- exact-format and constrained-generation success;
- answer termination and runaway-generation rate;
- repetition and degeneration metrics;
- held-out instruction loss;
- per-category success;
- multi-turn role consistency;
- concise-answer length distribution.

### Base retention

- complete `eval_core_v1` scorecard;
- global and per-cluster loss deltas;
- bits per byte;
- calibration and top-k accuracy deltas;
- fixed base-generation prompt deltas.

### Exactness and operations

- dataset manifest and all file hashes;
- deterministic split identity;
- loss-mask verification;
- checkpoint/resume equivalence;
- exact next sample after resume;
- remote recovery;
- dataset, teacher, code, optimizer, and schedule identities.

A chat-quality gain does not pass if the checkpoint violates correctness, provenance, or resume gates. A final numeric forgetting tolerance should be calibrated from the S0 pilot rather than invented after the complete run.

## Data-quality pipeline

Required stages:

1. License and provenance allowlist.
2. Scope filter: English, no deliberate code, no tool use, no advanced long-form reasoning.
3. Exact deduplication.
4. Near-duplicate clustering before split assignment.
5. Evaluation and benchmark decontamination.
6. Length and structural validation.
7. Language and toxicity screening.
8. Response-quality checks.
9. Tokenization and assistant-mask verification.
10. Immutable manifest publication.

For teacher-generated responses, generate multiple candidates where affordable. Apply deterministic filters first, then correctness verifiers for verifiable tasks, then a documented judge only for subjective tasks. Retain raw rejected generations for audit, but never silently add them to training.

## Mid-training policy

Reasoning-oriented mid-training is not the same experiment as chat SFT. It should be a separate branch from the frozen base checkpoint and should occur before chat SFT.

Proposed later comparison:

```text
base checkpoint
  ├─ direct SFT
  └─ continued/mid-training on educational or problem-solution text
       + original-distribution replay
       → identical SFT
```

Do not insert synthetic reasoning mid-training before the first S0 baseline, because that would prevent attributing changes to the chat stage. It is more valuable after a stronger base-model run, such as the same approximately-20M geometry on substantially more pretraining data or the first substantive approximately-100M geometry.

## Decisions still requiring explicit approval

1. Approve the S0 → S1 → S2 → S3 stage order.
2. Approve `smol-smoltalk` as the initial candidate source and authorize a pinned audit.
3. Approve the 15% pretraining replay default and 0/15/30 ablation set.
4. Approve the proposed 4M target-token budget.
5. Approve the plain-text chat template with no new tokens.
6. Approve one-conversation-per-sequence and no cross-example packing.
7. Approve full-parameter AdamW SFT and the bounded LR-selection protocol.
8. Approve rejecting ChatGPT/OpenAI outputs for training-data generation under current terms.
9. Select and approve a teacher only for S1 after current terms and provider conditions are captured.
10. Freeze exact evaluation gates after the bounded S0 pilot.

## Research anchors

- Hugging Face, SmolTalk and Smol-SmolTalk dataset cards and SmolLM2 report.
- Kotha and Liang, `Replaying pre-training data improves fine-tuning`, arXiv:2603.04964.
- Tang and Zhao, `SmartAD: Capacity-Aligned Agent Distillation for Small Language Models`, Findings of ACL 2026.
- Szep et al., `Complexity-aware fine-tuning`, Findings of EACL 2026.
- Li et al., `Rethinking On-Policy Distillation of Large Language Models`, arXiv:2604.13016.
- Xu et al., `Harnessing Negative Signals: Reinforcement Distillation from Teacher Data for LLM Reasoning`, ACL 2026.
- AMD, `Introducing ReasonLite-0.6B`, 2026.
- OpenAI Terms of Use and OpenAI Services Agreement effective 2026.
- DeepSeek Terms of Use, current 2026 version.
