# Approximately-20M Qualification Dataset Scope

_Last updated: 2026-08-04_

## Fixed profile

The first approximately-20M NVIDIA T4 qualification uses a separate finite
dataset with:

```text
target accepted source tokens: 10,000,000
minimum accepted source tokens: 9,000,000
hard maximum accepted source tokens: 11,000,000
context length: 2,048
sequences per prepared block: 16
target shard size: 8 MiB = 8,388,608 bytes
durable checkpoint cadence: 2,000,000 accepted source tokens
remote durability: required
trainer microbatch size: 1
trainer passes: 1
```

The 8 MiB target is specific to this small finite dataset so remote restore can
exercise multiple shards.  It does not replace the general 1 GiB large-corpus
default.

The 2M durability cadence reuses the cadence that passed the authenticated 10M
operational pilot.

## Build entry point

Use only the fail-closed qualification wrapper:

```bash
uv run --env-file .env python -m dataset.qualification_20m \
  --weights-file <approved-exact-weight-file.json> \
  --output-dir <new-qualification-dataset-directory> \
  --run-id 20m-qualification-dataset-001
```

The wrapper fixes and appends the token bounds, context length, block size,
shard size, and checkpoint cadence.  It rejects conflicting arguments and
rejects `--allow-local-only`.

Reader concurrency and bounded operational tuning may still be supplied when
they do not alter dataset identity.

## Target, minimum, and hard maximum

Documents are indivisible.  The producer never cuts a source document merely
to land exactly on 10,000,000 tokens.

The stopping contract is:

1. continue accepting complete documents while below 10,000,000;
2. normally stop when a document takes the total to at least 10,000,000;
3. never accept a document that would take the total above 11,000,000;
4. if the next indivisible document would exceed 11,000,000, completion below
   target is permitted only after at least 9,000,000 tokens are incorporated;
5. below 9,000,000, source exhaustion or the hard-maximum guard is a failure.

Examples:

```text
9,980,000 + 50,000 => accept and finish at 10,030,000
9,700,000 + 1,400,000 => reject; 11.1M breaches cap; 9.7M may complete
8,800,000 + 2,300,000 => reject and fail; current total is below minimum
```

These numbers count distinct accepted source tokens, not EOD markers, padding,
epochs, or repeated presentations.

## How cluster weights use the target

The approximately 10M accepted source tokens form the total material over which
the exact approved cluster weights are tracked.  For a hypothetical 20% cluster,
its cumulative target would be approximately 2M accepted source tokens.

Documents remain indivisible, so exact final cluster counts may differ slightly
from their ideal fractional quotas.  The deterministic deficit scheduler tracks
both cumulative and rolling mixture error.  Individual 32,768-target-token
optimizer blocks are not required to contain every cluster in exact global
proportion.

## What the completed manifest determines

The source-token profile is fixed now, while the following exact values are
outputs of the completed build:

- accepted source tokens;
- train and validation source tokens and documents;
- inserted EODs, stored tokens, and loss-bearing target tokens;
- complete ordered block-ID lists;
- number, sizes, and hashes of train and validation shards;
- exact one-pass optimizer-update count;
- configuration, schema, source-plan, local-manifest, and Drive-manifest hashes.

The exact training schedule and fixed validation slice are calculated from
these verified outputs.  Approximately 305 updates is only a planning estimate.

## Relationship to the accepted operational pilot

The previously accepted 10M pilot used 512 sequences per block and remains
immutable operational evidence.  It qualified source reading, exact mixture,
Drive durability, interruption/resume, verification, and idempotence.

It is not reused for training because its atomic-block contract would yield
only about ten optimizer updates.  The new dataset keeps the same pinned source,
tokenizer, accepted/excluded clusters, exact weights, and schema while changing
the prepared-block geometry to 16 sequences.

## Validation and lifecycle

The document-identity validation split remains deterministic.  Accepted cluster
IDs and exact cluster weights are unchanged; validation is not rebalanced or
oversampled.

The complete dataset is built, remotely durable, and verified before training.
The trainer performs exactly one pass and must stop at manifest exhaustion.
Producer/trainer overlap is a separate later operational qualification.
