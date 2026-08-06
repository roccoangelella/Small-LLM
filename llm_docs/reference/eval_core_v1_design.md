# `eval_core_v1` Design and Evaluation Integration

_Last updated: 2026-08-06_

## Decision summary

The user authorized implementation of a permanent, stratified pretraining evaluation set named `eval_core_v1` and a versioned checkpoint scorecard.

The evaluation work will be integrated with the existing qualitative prompt suite. A completed evaluation must still print and retain the model's answers to the existing fixed prompts; the intrinsic metrics do not replace those samples.

For the moment the project continues with the main GDN-2 hybrid architecture only. A matched all-attention or other architecture baseline is deferred until the project reaches larger model versions. The evaluation implementation must remain architecture-agnostic so it can be reused when those comparisons are eventually authorized.

## Why the size is defined by both documents and tokens

Document counts alone are unsafe because ClimbMix documents vary greatly in length. A small number of long documents could supply many tokens while giving poor independent coverage. Token counts alone are also unsafe because correlated tokens from a few documents can make uncertainty look smaller than it is.

`eval_core_v1` therefore has a per-cluster floor for both distinct documents and scored target tokens.

The design follows the useful current low-data evaluation pattern of keeping a smaller fast subset for intermediate checkpoints and a larger full set for final checkpoints. BabyLM's 2025 evaluation pipeline uses separate fast and full evaluations, while recent BabyLM baselines use a development set on the order of one million words. Our set is larger and explicitly stratified because we need reliable reporting across nineteen retained ClimbMix clusters rather than only one global validation number.

References:

- https://github.com/babylm/evaluation-pipeline-2025
- https://huggingface.co/BabyLM-community/babylm-baseline-10m-gpt-bert-causal-focus

## Frozen full-set target

There are nineteen retained clusters: `1-10` and `12-20`.

For every retained cluster, deterministic selection continues until both conditions are satisfied:

```text
minimum distinct source documents: 256
minimum scored target tokens: 131,072
```

`131,072` target tokens equal sixty-four complete 2,048-token evaluation windows.

The resulting full-set lower bounds are:

```text
minimum distinct document-cluster selections: 4,864
minimum scored target tokens: 2,490,368
minimum complete evaluation windows: 1,216
```

The actual total may be larger because both the document and token floors must pass. The manifest records the exact realized counts.

This is large enough to support document-level bootstrap intervals and useful per-cluster comparisons while remaining practical for final-checkpoint evaluation on a T4. The first implementation must measure real evaluation time and memory, but it may only increase the full set if uncertainty remains too wide; it must not silently shrink the frozen minimums.

## Frozen fast subset

The fast subset is a deterministic nested subset of the full set. For every retained cluster it contains at least:

```text
minimum distinct source documents: 32
minimum scored target tokens: 16,384
```

`16,384` target tokens equal eight complete 2,048-token evaluation windows.

The fast-set lower bounds are therefore:

```text
minimum distinct document-cluster selections: 608
minimum scored target tokens: 311,296
minimum complete evaluation windows: 152
```

The fast subset is intended for intermediate checkpoint monitoring. The full set is required for final logarithmic checkpoints and decisions about scaling.

## Source and leakage contract

`eval_core_v1` must be selected only from the existing deterministic document-level validation hash partition of the pinned Nemotron-ClimbMix revision:

```text
source revision: 5eaa64b9c0c85b7f56af01d7dffdb0795816b12b
accepted clusters: 1-10 and 12-20
excluded cluster: 11
context length: 2,048
semantic tokenizer: GPT-2 byte-level BPE IDs from source records
```

This is important because the current 100M dataset and the existing production pipeline already exclude that validation partition from training. Selecting from a new, broader partition after training began could accidentally include documents already consumed by the current run.

Every selected source document is identified durably by its immutable source-file identity, record position or equivalent stable record identity, cluster, document hash, source token count, and selection rank. The exact selected-document list is versioned and hashed.

Future dataset builders must preserve the existing validation-partition exclusion. A build fails closed if an `eval_core_v1` document is observed in a training output or if the pinned split/hash contract changes.

The old small prepared validation split remains part of the training record for continuity with the 10M and current 100M runs. It is not renamed or retroactively replaced. `eval_core_v1` is the larger static scientific evaluation set.

## Packing and attribution

Evaluation records use the same semantic token IDs, EOD handling, and context-plus-one next-token contract as training.

Selection and packing must retain cluster attribution. Cluster-level metrics are computed directly, then combined in two separate ways:

1. macro average across the nineteen clusters;
2. mixture-weighted average using the approved conditioned source-token weights.

These two aggregates must not be conflated.

Long documents may contribute multiple windows. The manifest records each document's scored-token contribution so document-macro and token-weighted summaries can both be produced. Confidence intervals are bootstrapped by source document, not by individual token.

## Versioned intrinsic scorecard

The first scorecard version must stream metrics without retaining full-vocabulary logits for the entire set. It records at least:

```text
negative log-likelihood in nats per target token
perplexity
bits per byte
top-1, top-5, and top-10 next-token accuracy
per-cluster token-weighted loss
cluster macro-average and mixture-weighted loss
worst-cluster loss
loss by sequence-position bucket
calibration summary, initially ECE
95% document-bootstrap intervals for global and per-cluster loss
evaluation target tokens, documents, windows, throughput, wall time, and peak VRAM
checkpoint, model, tokenizer, source, eval-set, and code identities
```

The implementation should keep accumulators modular so later frequency, document-length, generation-degeneration, standardized task, and inference-efficiency metrics can be added without changing `eval_core_v1` identity.

## Integration with the existing qualitative prompt suite

The existing `PROMPT_CASES` remain the single source of truth for qualitative prompts. Their wording, order, seed, and default sampling settings must not be duplicated in a second evaluator.

A unified post-pretraining evaluation flow will:

1. resolve and verify the selected native checkpoint;
2. reconstruct the model once;
3. run the fast or full `eval_core_v1` intrinsic scorecard;
4. run the existing qualitative prompt cases against the same loaded checkpoint;
5. print the model's prompt answers for human inspection;
6. save metrics, prompts, generated token IDs, decoded answers, sampling settings, and identities in one versioned result bundle.

The current prompt-only entry point remains supported. The integrated runner may call it as a component or share its checkpoint-loading and generation functions, but there must be no second prompt list.

The ordinary offline unit-test suite must test metric arithmetic, manifest validation, leakage rejection, deterministic selection, JSON schema, and prompt integration with toy models and tiny fixtures. It must not download a real checkpoint or run the multi-million-token evaluation during normal `unittest discover`. The real evaluation command is an explicit evidence-producing run whose console output still includes the model's answers.

## Architecture-comparison timing

The previous proposal to prepare and immediately train a matched 20M all-attention baseline is not authorized now.

Current rule:

```text
continue the main GDN-2 hybrid architecture through the present data-scaling work
no new sequence-mixer architecture run at the 20M stage
revisit matched architecture baselines when larger model versions are reached
```

This is a timing decision, not a claim that architecture baselines are scientifically unnecessary. The evaluator must not encode GDN-2-specific assumptions into generic scoring paths.

## Implementation order

```text
1. Implement deterministic eval document selection from the existing validation hash partition.
2. Produce and verify the full and nested fast manifests.
3. Add leakage checks binding future training outputs against the frozen document identities.
4. Implement streaming scorecard accumulators and document-bootstrap intervals.
5. Add a unified result schema and checkpoint-evaluation entry point.
6. Reuse the existing prompt suite in that flow and preserve printed answers.
7. Add toy-model and tiny-fixture unit/integration tests.
8. Benchmark fast and full evaluation time on the T4.
9. Evaluate the completed 10M checkpoint for a historical anchor, then the current 100M checkpoint when training finishes.
```
