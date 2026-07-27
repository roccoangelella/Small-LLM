# Small LLM Project Memory

## Project Goal

Build a sub-1B language model from random initialization that:

- Speaks and writes good English.
- Understands instructions and holds coherent conversations.
- Has useful reasoning capabilities.
- Uses modern small-LLM training techniques.
- Serves primarily as a learning and research project for an AI MSc student.

The current scope does **not** include coding capability. Coding can be added later as a separate extension.

---

## Macro Project Steps

### 1. Define the Goal

Specify what the model should be able to do.

Current target capabilities:

- Fluent English generation.
- Reading comprehension.
- Conversation.
- Instruction following.
- General knowledge.
- Basic and intermediate reasoning.
- Clear explanations.

Also define which capabilities are outside the initial scope.

### 2. Define the Resource Budget

Decide:

- Maximum model size.
- Available GPUs.
- Training time.
- Storage capacity.
- Maximum pretraining token budget.
- Target inference hardware.

These limits determine the feasible model and training plan.

### 3. Define Evaluation

Choose how progress will be measured before training begins.

Evaluation should cover:

- English fluency.
- Grammar and coherence.
- Reading comprehension.
- Instruction following.
- General knowledge.
- Logical and multi-step reasoning.
- Consistency and reliability.
- Inference speed and memory use.

Keep part of the evaluation data private and separate from all training data.

### 4. Design the Model

Choose the model architecture and approximate size.

The model should use a modern decoder-only Transformer design suitable for small language models.

Begin with a very small experimental model before choosing the final sub-1B model.

### 5. Adopt and Freeze the Tokenizer

Use the GPT-2 byte-level BPE tokenizer already used by Nemotron-ClimbMix rather than training a custom tokenizer.

The source documents already contain GPT-2 token IDs. The production corpus therefore reuses those IDs directly instead of decoding and retokenizing every accepted document.

Before model creation:

- Freeze the base GPT-2 vocabulary.
- Decide and append the small set of project special tokens needed for padding, document boundaries, chat, and instruction tuning.
- Save the tokenizer files, configuration, and hashes.
- Verify special-token behavior and basic source-token compatibility on a small test.

Tokenizer training is outside the current project scope unless a concrete limitation of GPT-2 tokenization is discovered later.

### 6. Collect and Prepare Pretraining Data

Use a manageable, publicly available English corpus that fits the project's storage and compute limits. Process it locally as a fixed, training-ready token corpus rather than attempting to manage a multi-terabyte mixture during training.

The current source is a roughly 25% cluster-stratified subset of **Nemotron-ClimbMix**, selected for broad English and general-knowledge coverage. NVIDIA's numeric `cluster_id` is the sole semantic/content-selection heuristic for the first production corpus.

Programming-oriented clusters are excluded or assigned no quota because coding capability is deferred. No document-level code-density, quality, topic, or LLM classifier is applied during this first production extraction. Incidental code, low-quality text, or broad-topic mismatches may therefore remain inside accepted clusters and are recorded as a known limitation.

The selected corpus is expected to be roughly 400 GB / 80–100B unique source tokens. The working training target is 2T token presentations, implying roughly 20–25 passes over that corpus. This is an experimental assumption to validate with held-out loss and downstream evaluations, not a claim that repeated epochs fully substitute for more unique data.

The production extraction should:

1. Stream Nemotron-ClimbMix source records.
2. Read each record's `cluster_id` and existing GPT-2 token IDs.
3. Accept records only while the configured cluster quota is unfilled.
4. Perform structural validation only, such as checking required fields, valid token-ID ranges, and non-empty token arrays.
5. Assign accepted documents reproducibly to train or validation data.
6. Write the existing token IDs directly into sharded binary training files with document or sequence-boundary indexes.
7. Maintain resumable checkpoints, per-cluster token/document counters, output hashes, and a manifest.
8. Check the final corpus for overlap with private evaluation sets where feasible.

The full corpus is not detokenized into plain-text JSON and is not passed through an LLM. Decoding a tiny number of records remains optional for debugging, but it is not a production filter or approval gate.

FineWeb-Edu, DCLM-Baseline, and selected Dolma 3 sources remain candidate alternatives for a later corpus comparison, not the current pretraining plan.

### Verified Nemotron-ClimbMix Cluster Map — 2026-07-27

The numeric `cluster_id` values must use NVIDIA's published CLIMB topic table. The earlier project map was wrong for all 20 numeric IDs and must not be reused. In particular: cluster 11 is software/programming, 15 is film/comics, 16 is sustainability/climate, 18 is cybersecurity/networking, and 20 is public safety/political history rather than Python code.

A bounded live check sampled five documents from every ID (100 documents total) and sent 20 fixed-schema Gemini reviews against the published map. It found 11 matches, 6 partial matches, and 3 broad-topic mismatches. This supports using NVIDIA's map as a broad selection heuristic while confirming that clusters are not perfectly pure. The evidence is stored in `cluster_map_validation.json` at the repository root.

The production decision is now to trust these cluster IDs without a second mandatory 50-document-per-cluster review, manual worksheet, document-level code filter, or LLM approval gate. Production selection may begin once the desired cluster quotas and output-shard settings are configured. The resulting corpus should be described as **programming-cluster-excluded**, not guaranteed code-free.

### 7. Pretrain the Base Model

Train the model to predict the next token across the prepared English corpus.

During pretraining, the model should learn:

- Vocabulary.
- Grammar.
- Writing style.
- General knowledge.
- Reading patterns.
- Basic reasoning patterns.
- Relationships between ideas and concepts.

Use the selected corpus as one auditable mixture; if additional sources are introduced later, define and document their mixture rather than training them as isolated stages.

### 8. Improve Reasoning Ability

Continue training with data selected specifically for reasoning.

Possible categories include:

- Mathematics.
- Logic problems.
- Step-by-step explanations.
- Scientific reasoning.
- Cause-and-effect questions.
- Comparisons and classification tasks.
- Synthetic reasoning examples generated by stronger models.
- Questions with automatically checkable answers.

Reasoning should be evaluated separately from English fluency.

The project should distinguish between:

- Reasoning learned during pretraining.
- Reasoning learned through distillation.
- Reasoning improved through later post-training.

### 9. Evaluate the Base Model

Freeze and evaluate the pretrained model before converting it into an assistant.

This establishes what the model learned from pretraining alone.

Test:

- Text continuation.
- English quality.
- Comprehension.
- Knowledge.
- Reasoning.
- Consistency.

### 10. Fine-Tune the Model as an Assistant

Train the base model on high-quality instruction and conversation examples.

This stage teaches the model to:

- Respond directly to users.
- Follow instructions.
- Answer questions.
- Explain ideas clearly.
- Maintain conversational structure.
- Format responses appropriately.

### 11. Improve Reasoning and Response Quality

Use modern post-training methods such as:

- Distillation from stronger models.
- Synthetic reasoning examples.
- Rejection sampling.
- Training on successful model outputs.
- Preference optimization.
- Verifiable-reward training for tasks with objectively checkable answers.

The goal is to improve reasoning without damaging English fluency or general capability.

### 12. Evaluate Every Major Version

Evaluate and compare:

- Initial base model.
- Reasoning-focused model.
- Instruction-tuned model.
- Distilled or preference-tuned model.
- Final model.

Check both improvements and regressions after every stage.

### 13. Optimize and Deploy

Prepare the final model for practical inference.

This includes:

- Reducing memory requirements.
- Improving generation speed.
- Choosing sensible generation settings.
- Packaging the tokenizer and model weights.
- Building a simple interface for testing and conversation.

### 14. Document the Project

Record:

- Model architecture.
- Tokenizer.
- Dataset sources.
- Data mixture.
- Selection and structural-validation process.
- Training settings.
- Compute usage.
- Evaluation results.
- Failed experiments.
- Known limitations.
- Ethical and licensing considerations.

---

## Current Dataset Decision

The initial pretraining corpus is a cluster-stratified subset of **Nemotron-ClimbMix**, targeting approximately 80–100B unique GPT-2 tokens and roughly 400 GB of local storage.

NVIDIA's published `cluster_id` map is the only content-selection heuristic for the first production extraction. Programming-oriented clusters receive no quota; accepted clusters are sampled according to the final configured mixture. There is no document-level code, quality, topic, or LLM filter. The corpus may contain incidental code and other cluster impurities, so it is programming-cluster-excluded rather than guaranteed code-free.

The source GPT-2 token IDs are retained directly and written into binary train/validation shards. The full corpus is not converted to plain-text JSON, Parquet, or another tokenizer during production.

The working training target remains 2T token presentations, while monitoring held-out loss and downstream evaluations for signs that repeated epochs are no longer beneficial.

---

## Current High-Level Training Flow

```text
Define goals and evaluation
→ Set the compute budget
→ Build and validate a small model
→ Adopt and freeze the GPT-2 tokenizer plus project special tokens
→ Configure Nemotron-ClimbMix cluster quotas
→ Stream selected records directly into binary train/validation token shards
→ Pretrain the English base model
→ Continue training for reasoning
→ Evaluate the base model
→ Instruction-tune it as an assistant
→ Improve it through distillation and verified reasoning data
→ Evaluate, optimize, document, and release
```

- Reasoning is learned from a mix of normal pretraining, clear worked examples, and practice on problems with correct answers.
- For this project, build strong English first, then add verified reasoning examples and later improve reasoning through guided practice.

---

## Current Open Decisions

- Final parameter count.
- Whether the 2T-token presentation target and repeated-epoch schedule are justified by evaluation results.
- Exact selected cluster list, per-cluster quotas, and final corpus size within Nemotron-ClimbMix.
- Exact project special-token set.
- Binary shard size, indexing scheme, and sequence-packing policy.
- Reasoning datasets and generation process.
- Teacher model used for distillation.
- Evaluation benchmark suite.
- Available compute and storage.
- Whether the final model will be released publicly.
