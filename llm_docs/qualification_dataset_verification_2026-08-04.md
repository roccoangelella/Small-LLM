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
stream-cache verifier.  That path correctly checked local shard existence,
byte sizes, SHA-256 checksums, block continuity, source-token attribution, and
Drive durability markers, but it did not forward the `full_scan` request into
token decoding.  Its schema-v2 report also used zero placeholders for document
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

## Gate

The qualification dataset is accepted for Kaggle packaging only after the new
command returns:

```json
{
  "passed": true,
  "complete": true,
  "full_scan": true,
  "problems": []
}
```

The exact `qualification_plan.json` should then be regenerated once more to
confirm that the manifest identities and 306-update schedule remain unchanged.
