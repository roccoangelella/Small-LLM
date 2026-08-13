# `eval_core_v1` runbook

_Last reviewed: 2026-08-13_

## Purpose

`eval_core_v1` is the frozen intrinsic evaluation used for scale decisions. The normal entrypoints discover or build the frozen corpus, verify it, download/verify the selected native checkpoint, reconstruct the model, stream metrics, and write one self-hashed JSON bundle.

A completed corpus contains:

```text
manifest.json
fast.bin
fast.records.jsonl
full.bin
full.records.jsonl
```

Kaggle automatically discovers an attached verified copy when present. Otherwise the evaluator uses its configured cache/build path and verifies before evaluation.

## Stable completed model artifacts

For completed `models/<run_id>/...` artifacts, use the stable transport wrapper:

```bash
python -m trainer.eval_entrypoint_model full \
  --repo-id <repo> \
  --run-id <run> \
  --pointer latest \
  --temperature 0 \
  --top-p 1 \
  --top-k 0 \
  --seed 17 \
  --samples-per-prompt 1 \
  --output-json artifacts/<run>_eval_full.json
```

Stable model artifacts verify native `local_manifest.json` and intentionally do not require live-run publication metadata.

## Live two-phase `run/...` checkpoints

For live/rolling checkpoints use:

```bash
python -m trainer.eval_entrypoint full \
  --repo-id <repo> \
  --run-id <run> \
  --pointer latest \
  --temperature 0 \
  --top-p 1 \
  --top-k 0 \
  --seed 17 \
  --samples-per-prompt 1 \
  --output-json artifacts/<run>_eval_full.json
```

Use `--pointer best` when the scientific question is explicitly validation-best selection rather than terminal/latest endpoint comparison.

## What `full` records

The result includes:

- NLL/loss and perplexity;
- bits per decoded target byte;
- top-1/5/10 next-token accuracy;
- ECE calibration and bins;
- per-cluster loss/perplexity;
- cluster macro and source-mixture-weighted loss;
- worst cluster;
- sequence-position bucket loss;
- global/per-cluster bootstrap intervals;
- wall time, throughput, and peak VRAM;
- prompt text/output/token IDs and decoding settings;
- checkpoint/model identity.

For scientific comparisons, require identical `eval_manifest_sha256`.

## Fast suite

Use `fast` for intermediate diagnostics; use `full` for endpoint/scale decisions. `--skip-prompts` is metric-only diagnostic mode, not a replacement for the project's separate qualitative qualification.

## Qualitative-protocol caveat

The full evaluator currently uses each prompt case's native generation budget and does not expose ADR 0025's global `max_new_tokens=32`. Therefore its intrinsic metrics are canonical, and prompt outputs from identically configured full runs are directly comparable, but those prompt outputs are **not the exact ADR-0025 canonical qualitative comparison**.

When the exact frozen qualitative comparison is required, also run [`post_pretraining_prompt_suite.md`](post_pretraining_prompt_suite.md) with the global 32-token cap.

## Current three-way reference

The accepted 20M/500M, 20M/2B, and 100M/2B full bundles use:

```text
eval manifest: aa7b6157e5f420dd53a99552685eaed01962ee45c23cbe438e1321a886422792
full targets: 3,095,181
```

Evidence and interpretation: [`../evidence/scaling/20m_500m_20m_2b_100m_2b_full_eval_2026-08-13.md`](../evidence/scaling/20m_500m_20m_2b_100m_2b_full_eval_2026-08-13.md).

## Offline verification/tests

The ordinary repository test suite remains network-free. Production corpus building or HF access should not occur in unit tests; entrypoint tests mock those boundaries.
