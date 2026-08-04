# Approximately-20M Qualification Dataset Scope

_Last updated: 2026-08-03_

## Decision

The user approved the accepted-source-token envelope for the finite dataset used by the first approximately-20M NVIDIA T4 training qualification:

```text
target accepted source tokens: 10,000,000
minimum accepted source tokens: 9,000,000
hard maximum accepted source tokens: 11,000,000
context length: 2,048
sequences per prepared block: 16
microbatch size: 1
```

This is a separate dataset build from the already accepted 10M operational pilot. The operational pilot remains immutable evidence with 512-sequence blocks. The training-qualification dataset must be built under a new run ID and output directory with explicit `--sequences-per-block 16`.

## Meaning of the envelope

The envelope controls the amount of distinct accepted source material prepared by the dataset producer. It does not specify epoch count or authorize repetition.

The producer aims for 10M accepted source tokens, must complete at or above 9M, and must not pass 11M except according to the existing whole-document stopping contract. The exact completed count and the exact number of training and validation target tokens are taken from the verified manifest.

The trainer's pass count remains a separate launch setting. The current protocol recommends one pass and forbids silent wraparound, but the user has not yet made repetition a separate explicit decision.

## Expected training geometry

At 16 sequences per block and context 2,048, a full training block contains approximately 32,768 target tokens. A roughly 10M-token training split would therefore provide about 305 optimizer updates. The exact count can differ because:

- the source-token envelope is not identical to stored target-token count;
- documents receive EOD boundaries;
- final sequences may contain padding;
- approximately 0.1% of accepted documents are assigned deterministically to validation;
- the final train and validation prepared blocks may contain fewer than 16 sequences.

Exact warmup, stable, decay, checkpoint, and evaluation positions must be derived from the verified manifest rather than from the approximate 305-update estimate.

## Validation implication

The frozen dataset policy assigns approximately 0.1% of accepted documents to validation. At a 10M source-token scale, the held-out split may be only around ten thousand source tokens, subject to document-size variance. The builder can finalize a partial validation block, so the build is not expected to fail merely because validation has fewer than 16 sequences. However, such a small validation sample will be noisy and is suitable primarily for functional qualification, not a strong model-quality estimate.

Before launch, the project must decide whether to:

1. retain the frozen 0.1% split and explicitly treat validation as a finite-value and checkpoint-usability smoke test; or
2. create an additional deterministic evaluation dataset or approved qualification-only split policy without changing the production corpus definition.

The existing production split probability must not be silently changed in the training command.

## Still open

The envelope decision does not yet settle:

- whether the trainer performs exactly one pass;
- whether the dataset is completed and verified before training or producer/trainer overlap is tested immediately;
- dataset durable checkpoint cadence for this new build;
- target shard size and remote-mirroring lifecycle;
- fixed validation slice and generation prompts;
- best-checkpoint definition;
- remote restore prefetch window;
- implementation of the additional metrics required by `20m_qualification_protocol.md`.

## Immediate implementation contract

The eventual dataset command must explicitly include:

```text
--target-tokens 10000000
--minimum-tokens 9000000
--maximum-tokens 11000000
--context-length 2048
--sequences-per-block 16
```

The trainer command must explicitly assert `--sequences-per-block 16`. A manifest geometry mismatch remains a hard failure.
