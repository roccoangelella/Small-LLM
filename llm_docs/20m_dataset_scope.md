# Approximately-20M Qualification Dataset Scope

_Last updated: 2026-08-04_

## Decision

The finite dataset used by the first approximately-20M NVIDIA T4 training qualification uses:

```text
target accepted source tokens: 10,000,000
minimum accepted source tokens: 9,000,000
hard maximum accepted source tokens: 11,000,000
context length: 2,048
sequences per prepared block: 16
microbatch size: 1
```

This is a separate dataset build from the already accepted 10M operational pilot. The operational pilot remains immutable evidence with 512-sequence blocks. The training-qualification dataset must use a new run ID and output directory with explicit `--sequences-per-block 16`.

## Precise meaning of target, minimum, and maximum

The phrase **9M-11M envelope** is only shorthand for a target plus two safety bounds. It does not mean that the producer arbitrarily chooses a size between nine and eleven million tokens.

Documents are indivisible: the producer either accepts a complete source document or does not accept it. It never cuts a document to hit exactly 10,000,000 tokens.

The stopping contract is:

1. continue accepting whole documents while the incorporated total is below 10,000,000;
2. normally stop as soon as an accepted whole document takes the total to at least 10,000,000;
3. never accept a document that would take the total above 11,000,000;
4. if the next indivisible document would exceed 11,000,000, the build may finalize below the 10M target only when at least 9,000,000 tokens have already been accepted;
5. if the source ends or the hard-maximum guard fires below 9,000,000, the build fails rather than producing an accepted qualification dataset.

Examples:

```text
current total 9,980,000 + next document 50,000
=> accept it and finish at 10,030,000

a current total 9,700,000 + next document 1,400,000
=> 11,100,000 would breach the hard maximum
=> refuse the document and permit completion at 9,700,000

a current total 8,800,000 + next document 2,300,000
=> the document would breach 11,000,000, but 8,800,000 is below the minimum
=> fail the build
```

Therefore, a normal run finishes slightly above 10M. The 9M lower bound exists only as a rare whole-document safety fallback. The verified manifest records the exact final count and whether the 10M target was reached or the hard-maximum guard caused early completion.

These numbers count distinct **accepted source tokens**. They do not count inserted EOD markers, padding, epochs, or repeated presentations.

## Training geometry

At 16 sequences per block and context 2,048, a full training block contains approximately 32,768 target tokens. A roughly 10M-token training split should provide about 305 optimizer updates. The exact count can differ because:

- source-token count is not identical to stored target-token count;
- documents receive EOD boundaries;
- final sequences may contain padding;
- approximately 0.1% of accepted documents are assigned deterministically to validation;
- final train and validation blocks may contain fewer than 16 sequences.

Exact warmup, stable, decay, checkpoint, and evaluation positions must be derived from the verified manifest rather than the approximate update estimate.

## Validation policy

The frozen dataset policy assigns documents to validation with a deterministic identity hash at probability 0.1%. The qualification leaves the accepted cluster IDs, cluster weights, and exact mixture scheduler untouched.

The project must not rebalance validation with new per-cluster quotas or move documents after the build. After completion it freezes the ordered validation block IDs, dataset and Drive-manifest hashes, token and document counts by cluster, and deterministic generation prompts.

At this scale the validation sample may be noisy and is treated primarily as a finite-value, checkpoint-usability, and gross-regression signal rather than a strong model-quality estimate.

## Dataset lifecycle and pass count

The dataset is completed, remotely durable, and fully verified before the trainer starts. The first qualification performs exactly one pass and forbids silent wraparound. Producer/trainer overlap remains a separate later operational test.

## Still open

The approved token limits and lifecycle do not yet settle:

- target shard size and durable dataset checkpoint cadence for this build;
- the exact frozen validation block list, which can only be recorded after completion;
- deterministic generation prompt contents;
- measured W&B warning and failure thresholds derived from the T4 preflight.

## Immediate implementation contract

The dataset command must explicitly include:

```text
--target-tokens 10000000
--minimum-tokens 9000000
--maximum-tokens 11000000
--context-length 2048
--sequences-per-block 16
```

The trainer command must explicitly assert `--sequences-per-block 16`. A manifest geometry mismatch remains a hard failure.
