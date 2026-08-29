# Post-pretraining qualitative and confidence suite

_Last reviewed: 2026-08-24_

## Canonical comparison protocol

ADR 0025 is authoritative. For every canonical full post-pretraining qualitative comparison, use the full prompt set and:

```text
temperature: 0
 top_p: 1
 top_k: 0
seed: 17
samples_per_prompt: 1
questions_only: false
max_new_tokens: 32
trace_top_tokens: 0
CUDA precision: fp16
```

The leading spaces above are formatting only; CLI flags are `--top-p 1 --top-k 0`.

Do **not** infer the canonical protocol from Python sampler defaults. Sampled decoding (`temperature=0.8`, `top_p=0.95`, `top_k=50`) remains useful supplementary evidence but is not the frozen cross-checkpoint comparison.

Do not add repetition penalties, no-repeat-ngram rules, or other decoding corrections to the canonical run.

## Installation

GDN-2 CUDA evaluation requires the production backend dependency:

```bash
python -m pip install "fla-core==0.5.2"
```

The project post-training dependencies also require PyTorch, `huggingface_hub`, and `tiktoken`.

## Live `run/...` checkpoint

For a live two-phase checkpoint repository, use the ordinary prompt-suite entrypoint and explicit run identity. ADR 0025 historically selects the validation-best pointer:

```bash
python -m trainer.post_pretraining_prompt_suite \
  --repo-id <repo> \
  --run-id <run> \
  --pointer best \
  --temperature 0 \
  --top-p 1 \
  --top-k 0 \
  --seed 17 \
  --samples-per-prompt 1 \
  --max-new-tokens 32 \
  --trace-top-tokens 0 \
  --output-json artifacts/<run>_prompts_greedy32.json
```

If a completed endpoint is intentionally being compared rather than its validation-best checkpoint, use `--pointer latest` and record that endpoint-selection choice.

## Stable `models/...` artifact

Completed human-facing artifacts use `trainer.post_pretraining_prompt_suite_model`. Stable artifacts preserve the terminal stable model, not the live run's validation-best pointer history, so use `latest` semantics and state that transport/selection difference explicitly:

```bash
python -m trainer.post_pretraining_prompt_suite_model \
  --repo-id <repo> \
  --run-id <run> \
  --pointer latest \
  --temperature 0 \
  --top-p 1 \
  --top-k 0 \
  --seed 17 \
  --samples-per-prompt 1 \
  --max-new-tokens 32 \
  --trace-top-tokens 0 \
  --output-json artifacts/<run>_prompts_greedy32.json
```

Stable artifacts verify their native `local_manifest.json`; they do not require the live two-phase `checkpoint_manifest.json` publication metadata.

## Why the 32-token cap matters

Greedy long-horizon generation can fall into high-probability loops. The 32-token cap makes the cross-checkpoint qualitative comparison less dominated by loop length while still exposing local continuation/factual behavior. Longer native-budget generation may be run separately as a degeneration diagnostic.

## Full-evaluator caveat

`trainer.eval_suite` currently embeds the same prompt definitions but does **not** expose a global `--max-new-tokens` cap. Consequently a full `eval_core_v1` bundle can contain useful mutually comparable greedy prompt output without exactly reproducing ADR 0025. Use this runbook's prompt-suite command when the exact canonical qualitative protocol is required.

## Teacher-forced held-out confidence diagnostic

Teacher-forced mode is deterministic and evaluates raw next-token logits against identity-matched schema-v2 held-out validation targets. It does not sample and therefore does not use temperature/top-k/top-p.

```bash
python -m trainer.post_pretraining_prompt_suite \
  --teacher-forced-validation \
  --output-json artifacts/teacher_forced_validation.json
```

Dataset identity remains fail-closed. Historical static datasets are matched by exact `drive_manifest.json` SHA-256. Modern incremental checkpoints are matched by their recorded `dataset_manifest_sha256`. In `auto` mode the evaluator first reuses an identity-matched local Kaggle dataset/cache; if none is present for a modern incremental run, it reconstructs the stable consumer manifest from the HF run contract/frontier, verifies that manifest hash against the checkpoint, and downloads only the frozen validation shards into the Kaggle working cache. The automatic remote path uses `SMALL_LLM_HF_DATASET_BUCKET_ID` when set, otherwise `<SMALL_LLM_HF_REPO_ID>-datasets`, with `HF_TOKEN` for private access.

It records true-token probability/rank, top-1/top-5 predictions, entropy, top-k rates, representative low-probability targets, and high-confidence errors over the frozen validation sample. Reports include `dataset_manifest_sha256` and retain `drive_manifest_sha256` when the selected legacy dataset has one. It complements `eval_core_v1`; it does not replace it.

## Interpretation rule

A pretrained base model is not an instruction-tuned assistant. Treat these prompts as narrow diagnostics of continuation, relation binding, factual retrieval, repetition, and termination. Always record decoding settings with any qualitative claim. In particular, the 100M/2B sampled run answered `Paris` while the later greedy run answered `France`; neither should overwrite the other.
