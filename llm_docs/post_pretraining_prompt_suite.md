# Post-Pretraining Qualitative Prompt Suite

_Last updated: 2026-08-05_

## Decision and scope

The repository includes a small qualitative prompt suite to run after a base-model pretraining campaign completes. This decision does **not** assert that the current qualification or eventual substantive pretraining has already completed; it defines the inspection procedure to use when a completed base checkpoint exists.

The suite is repository-native because Hugging Face stores the project's verified joint checkpoints rather than a Transformers-format export. It:

1. resolves `run/<run_id>/best.json` from the configured private Hugging Face repository;
2. downloads the referenced custom checkpoint tree;
3. verifies `local_manifest.json` and the complete pointer-bound publication manifest before loading state;
4. reconstructs `SmallLLM` from the model configuration embedded in new trainer checkpoints;
5. loads the native model weights;
6. tokenizes prompts with the GPT-2 byte-level BPE identity used by the project;
7. prints seeded base-model continuations and can save the complete result as JSON.

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

## Standard run

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

## Interpretation

The suite answers a narrow question: does the completed base model produce locally coherent English continuations and show recognizable learned knowledge or patterns?

It does not replace:

- held-out loss and perplexity;
- domain-level validation analysis;
- standardized base-model benchmarks;
- memorization and contamination checks;
- context-retrieval tests;
- architecture-matched comparisons;
- later supervised instruction tuning and preference evaluation.

The JSON output should be retained with the final evaluation artifacts so the same prompts, seeds, and sampling settings can be compared across checkpoints and later post-training stages.
