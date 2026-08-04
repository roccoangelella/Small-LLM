# 20M qualification dataset verification — 2026-08-04

## Observed completed dataset

The fixed 10M/16-sequence qualification build completed on the VPS with:

```text
run ID: 20m-qualification-dataset-001
accepted source tokens: 10,000,662
train source tokens: 9,991,872
validation source tokens: 8,790
train shards: 6
validation shards: 1
train sequences: 4,886
validation sequences: 5
train blocks / one-pass updates: 306
manifest SHA-256: 1e5ee8f372b77b6728288610dbe7cce74d833be21e53d1538bc5a890229b18bb
Drive manifest SHA-256: fbb29ee0d0102658e1274e39d6647cf56a6dcb685e0f566b1736847dcc4fbe84
```

The generated `qualification_plan.json` fixes the one-pass WSD schedule to:

```text
warmup: 16 updates / 524,288 target tokens
stable: 228 updates / 7,471,104 target tokens
decay: 62 updates / 2,011,136 target tokens
total: 306 updates / 10,006,528 target tokens
minimum LR ratio: 0.1
validation blocks: 1
```

## Verification gap found

The general `dataset.main verify --full-scan` command selected the schema-v2
stream-cache verifier. That path correctly checked local shard existence,
byte sizes, SHA-256 checksums, block continuity, source-token attribution, and
Drive durability markers, but it did not forward the `full_scan` request into
token decoding. Its schema-v2 report also used zero placeholders for document
and inserted-EOD counters that are not available in the schema-v2 manifest.

The successful earlier output therefore remains valid structural and checksum
evidence, but it is not by itself literal token-by-token scan evidence.

## Decision

Use a dedicated fail-closed qualification verifier rather than changing the
legacy 90B verifier path immediately:

```bash
uv run python -m dataset.qualification_20m_verify \
  --dataset-dir /data/small-llm-20m-qualification-001
```

The dedicated verifier:

1. runs the existing schema-v2 structural/hash verifier;
2. requires the fixed 20M qualification profile and completion state;
3. independently decodes every little-endian `uint16` token in every shard;
4. rejects token IDs outside the GPT-2 range `0..50256`;
5. checks per-shard token, sequence, and block geometry;
6. aggregates real per-cluster source-token totals;
7. compares aggregate train attribution with scheduler totals;
8. reports unavailable document/EOD counters explicitly rather than as real
   zero measurements;
9. exits nonzero on any problem.

Focused offline tests cover a valid complete scan, an out-of-range token whose
checksum still matches its manifest, and a sequence-geometry mismatch.

## Executed evidence

The focused verifier suite was executed on the VPS and passed:

```text
3 passed
```

The dedicated verifier was then executed against the completed qualification
dataset. It returned exit code `0` and the following acceptance fields:

```text
passed: true
complete: true
full_scan: true
problems: []
shards scanned: 7
stored uint16 tokens scanned: 10,021,659
train stored tokens: 10,011,414
validation stored tokens: 10,245
train sequences: 4,886
validation sequences: 5
accepted source tokens: 10,000,662
```

The full scan aggregated the exact source-token attribution:

```text
cluster 4:    530,262
cluster 6:  2,219,982
cluster 7:  1,991,306
cluster 12: 2,340,422
cluster 16: 1,645,691
cluster 17:   925,329
cluster 18:   347,670
total:      10,000,662
```

`accepted_document_count` and `inserted_eod_count` remain explicitly marked
unavailable because the schema-v2 manifest does not preserve those exact
counters. They are not represented as zero measurements.

The exact `qualification_plan.json` was regenerated after the full scan. Its
manifest SHA-256, Drive manifest SHA-256, seven-shard identity, 306-update
one-pass schedule, and warmup/stable/decay horizons remained unchanged.

## Gate result

The qualification dataset verification gate is **passed**. The dataset is
accepted for private Kaggle packaging and exact-commit trainer qualification.

This does not authorize the complete one-pass training segment by itself. The
remaining gates are the complete exact-commit offline suite, corrected T4
harness, 20-successful-update W&B preflight and threshold freeze, then local and
remote recovery qualification.
