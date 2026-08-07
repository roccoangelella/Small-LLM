# `eval_core_v1` Runbook

_Last updated: 2026-08-07_

## Normal user-facing workflow

The normal complete evaluator is now self-provisioning. `small-llm-eval` ensures the frozen `eval_core_v1` corpus exists, builds it when absent, verifies it, and only then downloads/verifies the selected checkpoint and evaluates it.

Use `uv run --extra post-training` so the post-training dependencies and console entry point are provisioned as part of the invocation.

Canonical final evaluation example:

```bash
uv run --extra post-training small-llm-eval full \
  --repo-id roccoangelella/small-llm-20m-qualification \
  --run-id 20m-100m-dataset-001 \
  --pointer best \
  --output-json artifacts/100m_eval_full_best.json
```

No `--eval-dir` is required in the normal workflow.

Default eval-corpus cache location:

```text
Kaggle:              /kaggle/working/eval_core_v1
other environments: artifacts/eval_core_v1
environment override: SMALL_LLM_EVAL_DIR
CLI override:         --eval-dir /custom/path
```

If the selected directory does not exist, the evaluator builds the frozen corpus. If it already exists, the evaluator reuses it. In both cases verification runs before model evaluation. An invalid existing corpus fails closed and is never silently overwritten.

## Installed commands

The repository exposes:

```text
small-llm-eval        self-provision eval_core_v1 and run the complete evaluator
small-llm-eval-data   explicit low-level build or verify operations
```

The low-level data command remains available for debugging, corpus publication, and explicit reproducibility checks, but it is not a prerequisite for the normal evaluator.

## What the frozen corpus contains

`eval_core_v1` is generated evaluation data, not source-controlled repository content. Its builder reads the pinned Nemotron-ClimbMix source through deterministic HTTP byte ranges and selects only records assigned to the already-frozen validation hash partition.

A completed corpus contains:

```text
manifest.json
fast.bin
fast.records.jsonl
full.bin
full.records.jsonl
```

The builder refuses to replace an existing output directory. It writes through a temporary sibling directory and publishes the complete tree only after both suite quotas pass and every file hash is computed.

The fast suite is nested inside the full suite. The production builder does not expose quota or split overrides; smaller settings exist only as Python injection points for offline tests.

## Explicit low-level corpus operations

Build manually only when needed:

```bash
uv run --extra post-training small-llm-eval-data build \
  --output-dir /custom/path/eval_core_v1
```

Verify manually:

```bash
uv run --extra post-training small-llm-eval-data verify \
  --eval-dir /custom/path/eval_core_v1
```

Verification fails closed on manifest/hash drift, source identity drift, missing files, binary/JSONL geometry mismatch, non-contiguous sequence indexes, invalid metadata, aggregate-count mismatch, missed per-cluster floors, or a fast set that is not a document-level subset of the full set.

## Checkpoint selection

The evaluator uses the existing verified private-Hugging-Face checkpoint flow.

For canonical final evaluation, state the model identity explicitly:

```text
--repo-id <HF repository>
--run-id <run ID>
--pointer best
```

`best` is the validation-selected checkpoint and remains the default pointer. `--pointer latest` is reserved for explicit endpoint diagnostics.

Environment variables remain supported:

```bash
export HF_TOKEN='...'
export SMALL_LLM_HF_REPO_ID='owner/private-checkpoint-repository'
export SMALL_LLM_RUN_ID='pretraining-run-id'
```

## Fast suite

Use `fast` for intermediate checkpoints and routine comparisons:

```bash
uv run --extra post-training small-llm-eval fast \
  --repo-id owner/checkpoints \
  --run-id run-id \
  --pointer best \
  --output-json artifacts/eval-fast.json
```

## Full suite

Use `full` for final checkpoints and scale decisions:

```bash
uv run --extra post-training small-llm-eval full \
  --repo-id owner/checkpoints \
  --run-id run-id \
  --pointer best \
  --output-json artifacts/eval-full.json
```

The complete run:

1. resolves the eval-core cache location;
2. builds the frozen corpus if absent;
3. verifies `eval_core_v1`;
4. downloads and verifies the selected native checkpoint;
5. reconstructs the model once;
6. streams the intrinsic scorecard;
7. prints the complete qualitative prompt suite and every generated continuation;
8. writes one self-hashed JSON bundle containing metrics, prompts, generated token IDs, decoded answers, sampling settings, checkpoint identity, and model geometry.

## Useful options

T4 defaults resolve automatically to CUDA FP16 and batch size one. A larger evaluation batch can be attempted explicitly after measuring memory:

```bash
uv run --extra post-training small-llm-eval fast \
  --repo-id owner/checkpoints \
  --run-id run-id \
  --output-json artifacts/eval-fast.json \
  --batch-size 2
```

Question-only qualitative output:

```bash
uv run --extra post-training small-llm-eval fast \
  --repo-id owner/checkpoints \
  --run-id run-id \
  --output-json artifacts/eval-fast-questions.json \
  --questions-only
```

Greedy prompt generation:

```bash
uv run --extra post-training small-llm-eval full \
  --repo-id owner/checkpoints \
  --run-id run-id \
  --output-json artifacts/eval-full-greedy.json \
  --temperature 0 \
  --top-p 1 \
  --top-k 0
```

`--skip-prompts` exists for metric-only diagnostics. It is not the normal complete-suite path because the project decision requires retaining and printing the model's answers.

## Recorded metrics

The result schema records:

```text
negative log-likelihood and perplexity
bits per decoded target byte
top-1, top-5, and top-10 next-token accuracy
ECE calibration and bin summaries
per-cluster losses and perplexities
cluster macro and exact-mixture-weighted loss
worst-cluster loss
loss by sequence-position bucket
document-bootstrap 95% intervals globally and per cluster
wall time, throughput, and peak allocated VRAM
```

The exact ClimbMix source-token totals and approved weights-file SHA-256 are embedded in the evaluation manifest and used for the mixture-weighted score.

## Offline repository tests

The ordinary test suite remains network-free:

```bash
uv run --extra model python -m unittest discover -v
```

The self-provisioning entry-point tests mock corpus construction/verification. Normal offline unit tests do not contact Hugging Face or build the multi-million-token production corpus.
