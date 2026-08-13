# `eval_core_v1` design

_Last reviewed: 2026-08-13_

## Purpose

`eval_core_v1` is the permanent frozen scientific pretraining evaluation set. It is architecture-agnostic and is kept separate from the small per-run validation sample used for training monitoring/checkpoint selection.

Intrinsic metrics do not replace qualitative model-output inspection; the two answer different questions.

## Source and leakage contract

The corpus is selected only from the existing deterministic validation partition of the pinned source:

```text
source: nvidia/Nemotron-ClimbMix
revision: 5eaa64b9c0c85b7f56af01d7dffdb0795816b12b
retained clusters: 1-10 and 12-20
excluded cluster: 11
context: 2,048
semantic tokenizer: source GPT-2 token IDs
```

Selected documents are durably identified and excluded from training by the same split contract. A future data pipeline must fail closed rather than allow an `eval_core_v1` document into training.

## Frozen full selection rule

For each of the nineteen retained clusters, deterministic selection continues until both floors pass:

```text
minimum distinct documents: 256
minimum scored target tokens: 131,072
```

The realized current full corpus used by the completed scaling comparison is:

```text
eval_manifest_sha256: aa7b6157e5f420dd53a99552685eaed01962ee45c23cbe438e1321a886422792
documents: 4,904
sequences: 5,177
scored target tokens: 3,095,181
decoded target bytes: 14,306,073
```

Do not silently shrink or replace this corpus when comparing checkpoints.

## Frozen fast subset

The fast suite is a deterministic document-level subset of full. Per retained cluster it has floors of 32 distinct documents and 16,384 scored target tokens. It exists for intermediate diagnostics; final scale decisions use full.

## Packing

Evaluation uses the same semantic IDs, EOD policy, and context+1 next-token geometry as training. Cluster attribution is preserved through packing so domain metrics are computed directly from scored targets.

## Metrics

The scorecard records:

- token NLL/loss and perplexity;
- bits per decoded target byte;
- top-1/5/10 next-token accuracy;
- ECE calibration bins;
- per-cluster loss/perplexity;
- equal-cluster macro loss;
- exact source-mixture-weighted cluster loss;
- worst-cluster loss;
- sequence-position bucket loss;
- document-bootstrap 95% intervals globally and per cluster;
- evaluation wall time, throughput, and peak VRAM.

Macro, source-mixture-weighted, and ordinary token-weighted global loss use different weighting schemes and may move in different directions. Report them separately rather than treating one as an error when they disagree.

## Reproducibility

The manifest binds source identity, selected documents, tokenizer/split/packing policy, files, hashes, counts, cluster floors, and approved mixture-weight identity. `fast` must remain a subset of `full`. Verification fails closed on hash, geometry, source, count, or selection drift.

Use [`../runbooks/eval_core_v1_runbook.md`](../runbooks/eval_core_v1_runbook.md) for current stable/live checkpoint commands and [`training_and_evaluation.md`](training_and_evaluation.md) for the scorecard interpretation boundary.
