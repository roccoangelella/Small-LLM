# Post-Pretraining Model-Output Suite

_Last updated: 2026-08-07_

## Decision and scope

The repository includes a small model-output suite to run after a base-model pretraining campaign completes. This decision does **not** assert that the current qualification or eventual substantive pretraining has already completed; it defines the inspection procedure to use when a completed base checkpoint exists.

The suite is repository-native because Hugging Face stores the project's verified joint checkpoints rather than a Transformers-format export. It:

1. resolves `run/<run_id>/best.json` from the configured private Hugging Face repository;
2. downloads the referenced custom checkpoint tree;
3. verifies `local_manifest.json` and the complete pointer-bound publication manifest before loading state;
4. reconstructs `SmallLLM` from the model configuration embedded in new trainer checkpoints;
5. loads the native model weights;
6. tokenizes prompts with the GPT-2 byte-level BPE identity used by the project;
7. prints seeded base-model continuations and can save the complete result as JSON;
8. can instead run a teacher-forced confidence diagnostic on identity-matched held-out validation text.

The authoritative entrypoint is:

```text
python -m trainer.post_pretraining_prompt_suite
```

## Best-checkpoint definition

For remote publication, **best** means the published checkpoint with the lowest held-out validation loss.

The stored higher-is-better comparison metric is:

```text
best metric = -validation_loss
```

After a checkpoint tree and `latest.json` have been uploaded and verified, the trainer may move `best.json` to that same immutable `.../last` snapshot. It includes the checkpoint manifest in the best pointer. The best selection therefore adds only a small pointer write and does not upload a duplicate checkpoint tree under a second prefix.

A resumed run reads the existing remote `best.json` metric before publishing further checkpoints. It therefore cannot silently replace an earlier best checkpoint merely because the trainer process restarted.

A checkpoint without a validation result can still update `latest.json`, but it cannot update `best.json`.

Legacy best pointers produced directly by `TwoPhaseCheckpointPublisher` and pointing to an immutable `.../best` tree remain accepted by the prompt suite.

## Self-describing checkpoints

New trainer checkpoints include the complete native model configuration alongside the model state. The prompt suite therefore reconstructs the exact geometry and architecture without relying on a manually repeated model-size flag.

Older checkpoints remain loadable by the trainer. For qualitative generation from an older checkpoint that predates the embedded configuration, supply an explicit JSON object through:

```text
--model-config-json /path/to/model_config.json
```

## Installation

From the repository root:

```bash
python -m pip install -e ".[post-training]"
```

The post-training extra adds PyTorch, `huggingface_hub`, and `tiktoken`. Dataset-only environments remain lightweight.

## Authentication and repository selection

The default environment variables are:

```bash
export HF_TOKEN='...'
export SMALL_LLM_HF_REPO_ID='owner/private-checkpoint-repository'
export SMALL_LLM_RUN_ID='pretraining-run-id'
```

`SMALL_LLM_RUN_ID` is optional when the repository contains exactly one matching `best.json` pointer. It should be set explicitly when the checkpoint repository contains multiple runs.

Only run the suite against a trusted project-controlled repository. The checkpoint tree and its hashes are verified before `trainer_state.pkl` is unpickled, but Python pickle is not a safe interchange format for untrusted third-party artifacts.

## Standard qualitative run

```bash
python -m trainer.post_pretraining_prompt_suite \
  --output-json artifacts/post_pretraining_prompt_suite.json
```

Defaults:

```text
pointer: best
sampling: temperature 0.8, top-p 0.95, top-k 50
seed: 17
samples per prompt: 1
CUDA precision: FP16
automatic CPU precision: FP32
```

The suite prints the selected repository, run, checkpoint, remote metric, model geometry, training step, consumed-token count, prompt text, and generated continuation.

## Faster question-only run

```bash
python -m trainer.post_pretraining_prompt_suite \
  --questions-only \
  --output-json artifacts/post_pretraining_questions.json
```

The question set contains stable English general-knowledge prompts, including capitals, astronomy, literature, basic science, geography, the human body, calendar knowledge, and simple arithmetic. Translation prompts are intentionally excluded because the approved data and initial capability target are English general knowledge.

## Prompt categories

The default suite contains:

- free prose and story continuation;
- encyclopedia-style continuation;
- a short dialogue continuation;
- structured pattern completion;
- a simple few-shot sentiment pattern;
- twelve simple general-knowledge questions in `Question: ... Answer:` form.

These are qualitative base-model probes, not instruction-following or alignment tests. A pretrained base model may continue the surrounding format, add further questions, contradict itself, repeat, or produce plausible but incorrect statements. Those behaviors are evidence about the base checkpoint rather than suite failures.

## Useful variants

Greedy deterministic decoding:

```bash
python -m trainer.post_pretraining_prompt_suite \
  --temperature 0 \
  --top-p 1 \
  --top-k 0
```

Two samples per prompt:

```bash
python -m trainer.post_pretraining_prompt_suite --samples-per-prompt 2
```

A short smoke run:

```bash
python -m trainer.post_pretraining_prompt_suite --max-cases 4
```

Inspect the latest checkpoint rather than the validation-selected best checkpoint:

```bash
python -m trainer.post_pretraining_prompt_suite --pointer latest
```

The best pointer remains the default for final qualitative inspection.

## Short-generation diagnostic

During the 20M-model / approximately-100M-token qualification, greedy generation exposed high-probability repetitive loops in long continuations. Keep the existing long prompt suite because it is useful for detecting degeneration, but also use a short-generation diagnostic to separate local next-token quality from long-horizon repetition.

The canonical short diagnostic is:

```bash
python -m trainer.post_pretraining_prompt_suite \
  --max-cases 6 \
  --temperature 0 \
  --top-p 1 \
  --top-k 0 \
  --max-new-tokens 32 \
  --trace-top-tokens 5 \
  --output-json artifacts/prompts_short_greedy_top5.json
```

`--max-new-tokens N` is a global cap: each case uses `min(case.max_new_tokens, N)`. Omitting it preserves the existing per-prompt budgets. `--trace-top-tokens K` is disabled by default; when positive, the suite prints and stores the chosen token's raw probability plus the top-K raw next-token candidates at every generated step.

The top-token probabilities are computed from the model's raw next-token logits before temperature scaling, top-k filtering, or top-p filtering. They are diagnostic evidence about the learned distribution, not a decoding modification. The JSON result includes the run sampling/diagnostic settings, each case's effective generation budget, and a per-step token trace when tracing is enabled.

Do not apply repetition penalties, no-repeat-ngram rules, or other decoding corrections in the canonical diagnostic. The long prompt suite remains useful for detecting degeneration; the short diagnostic is complementary rather than a replacement.

## Teacher-forced held-out confidence diagnostic

The greedy trace showed two different regimes: low confidence at semantic branching points and increasing confidence after the model enters repetitive or template-like trajectories. To distinguish broad uncertainty from confidently wrong predictions, the suite now supports teacher-forced analysis against the same schema-v2 held-out next-token targets used by validation loss.

Run it in the same Kaggle notebook with the accepted dataset attached:

```bash
python -m trainer.post_pretraining_prompt_suite \
  --teacher-forced-validation \
  --output-json artifacts/teacher_forced_validation.json
```

`--teacher-forced-validation` switches the suite from free generation to deterministic teacher-forced analysis. With no value, it scans the attached Kaggle inputs and requires exactly one dataset whose `drive_manifest.json` hash matches the verified checkpoint. An explicit dataset root can be supplied as the optional value:

```bash
python -m trainer.post_pretraining_prompt_suite \
  --teacher-forced-validation /path/to/dataset \
  --output-json artifacts/teacher_forced_validation.json
```

The canonical diagnostic uses the first 4,096 active validation targets in deterministic shard/sequence order. Inference remains one sequence at a time, and distribution metrics are computed in 256-position chunks so full-vocabulary analysis remains bounded on the T4. The dataset reader uses the same schema-v2 `context+1` contract as training: `input_ids=tokens[:-1]` and `labels=tokens[1:]`.

For every measured target position the JSON records:

- a short decoded context tail;
- the true held-out token and its raw probability;
- the true token's rank in the complete vocabulary;
- the raw top-1 token and probability;
- the raw top-5 candidates and their probabilities;
- top-5 probability mass;
- next-token entropy in nats.

The aggregate report records sampled loss/perplexity, mean and median true-token probability, mean and median top-1 probability, top-1 accuracy, true-token top-5/top-10/top-100 rates, median true-token rank, mean entropy, mean top-5 mass, the fraction of positions whose top-1 probability is below 0.1, and the fraction that are wrong despite top-1 probability at least 0.5. It also prints representative lowest-true-probability and highest-confidence-wrong positions.

This mode is not a decoding test: no temperature, top-k, top-p, repetition penalty, or sampling operation is applied to the measured distribution. Its purpose is to explain *how* a given validation perplexity arises and to provide a deterministic confidence/rank baseline for later model scales and post-training stages.

## Interpretation

The qualitative suite answers a narrow question: does the completed base model produce locally coherent English continuations and show recognizable learned knowledge or patterns? The teacher-forced mode complements that by asking whether the correct held-out next token is present near the top of the learned distribution and how confident the model is when it is wrong.

The suite does not replace:

- held-out loss and perplexity over the canonical validation protocol;
- domain-level validation analysis;
- standardized base-model benchmarks;
- memorization and contamination checks;
- context-retrieval tests;
- architecture-matched comparisons;
- later supervised instruction tuning and preference evaluation.

The JSON output should be retained with the final evaluation artifacts so the same prompts, seeds, sampling settings, and teacher-forced metrics can be compared across checkpoints and later post-training stages.
