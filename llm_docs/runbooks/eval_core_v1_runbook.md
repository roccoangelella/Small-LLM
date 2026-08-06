# `eval_core_v1` Runbook

_Last updated: 2026-08-06_

## Installed commands

Install the post-training dependencies and console entry points:

```bash
uv sync --locked --extra post-training
```

The repository exposes two commands:

```text
small-llm-eval-data   build or verify the frozen evaluation corpus
small-llm-eval        run intrinsic metrics and the existing qualitative prompts
```

The existing module forms remain equivalent:

```text
python -m dataset.eval_core
python -m trainer.eval_suite
```

## Build the frozen corpus

The build reads the pinned Nemotron-ClimbMix source through deterministic HTTP byte ranges and selects only records assigned to the already-frozen validation hash partition.

```bash
small-llm-eval-data build \
  --output-dir /data/eval_core_v1
```

The command creates one immutable directory containing:

```text
manifest.json
fast.bin
fast.records.jsonl
full.bin
full.records.jsonl
```

The builder refuses to replace an existing output directory. It writes through a temporary sibling directory and publishes the complete tree only after both suite quotas pass and every file hash is computed.

The fast suite is nested inside the full suite. The production CLI does not expose quota or split overrides; smaller settings exist only as Python injection points for offline tests.

`--max-work-items` is a diagnostic source-scan bound. It is not expected to complete a production build unless the frozen quotas happen to pass within that bound.

## Verify the corpus

Run this after copying, downloading, or attaching the evaluation directory:

```bash
small-llm-eval-data verify \
  --eval-dir /data/eval_core_v1
```

Verification fails closed on:

- manifest self-hash mismatch;
- source revision, context, cluster, or schema drift;
- missing files;
- binary or JSONL hash mismatch;
- binary geometry mismatch;
- non-contiguous sequence indexes;
- invalid document, cluster, or valid-target metadata;
- aggregate count mismatch;
- a missed per-cluster document or target-token floor;
- a fast set that is not a document-level subset of the full set.

## Checkpoint environment

The complete evaluator reuses the existing verified private-Hugging-Face checkpoint flow and its environment variables:

```bash
export HF_TOKEN='...'
export SMALL_LLM_HF_REPO_ID='owner/private-checkpoint-repository'
export SMALL_LLM_RUN_ID='pretraining-run-id'
```

`SMALL_LLM_RUN_ID` may be omitted only when the repository contains exactly one matching pointer. The best validation-selected checkpoint is used by default; pass `--pointer latest` to inspect the latest published checkpoint instead.

## Fast suite

Use the fast suite for intermediate checkpoints and routine comparisons:

```bash
small-llm-eval fast \
  --eval-dir /data/eval_core_v1 \
  --output-json artifacts/eval-fast.json
```

The default run:

1. verifies `eval_core_v1`;
2. downloads and verifies the selected native checkpoint;
3. reconstructs the model once;
4. streams the fast intrinsic scorecard;
5. prints the complete existing qualitative prompt suite and every generated continuation;
6. writes one self-hashed JSON bundle containing metrics, prompts, generated token IDs, decoded answers, sampling settings, checkpoint identity, and model geometry.

## Full suite

Use the full suite for final logarithmic checkpoints and scale decisions:

```bash
small-llm-eval full \
  --eval-dir /data/eval_core_v1 \
  --output-json artifacts/eval-full.json
```

The interface and output schema are the same as the fast suite. Only the evaluated corpus changes.

## Useful options

T4 defaults resolve automatically to CUDA FP16 and batch size one. A larger evaluation batch can be attempted explicitly after measuring memory:

```bash
small-llm-eval fast \
  --eval-dir /data/eval_core_v1 \
  --output-json artifacts/eval-fast.json \
  --batch-size 2
```

Question-only qualitative output:

```bash
small-llm-eval fast \
  --eval-dir /data/eval_core_v1 \
  --output-json artifacts/eval-fast-questions.json \
  --questions-only
```

Greedy prompt generation:

```bash
small-llm-eval full \
  --eval-dir /data/eval_core_v1 \
  --output-json artifacts/eval-full-greedy.json \
  --temperature 0 \
  --top-p 1 \
  --top-k 0
```

`--skip-prompts` exists for metric-only diagnostics. It is not the normal complete-suite path because the project decision requires retaining and printing the model's answers.

## Recorded metrics

The first result schema records:

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

The evaluation tests use injected records, tiny binary fixtures, and toy models. They do not contact Hugging Face, download a checkpoint, or run the multi-million-token corpus during normal unit testing.
