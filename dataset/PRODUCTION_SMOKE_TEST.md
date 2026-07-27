# Production pipeline smoke test — 2026-07-27

This was a bounded live read against the real pinned source. It did not launch
the 90B-token build.

## Source and bounds

- Repository: `nvidia/Nemotron-ClimbMix`
- Revision: `5eaa64b9c0c85b7f56af01d7dffdb0795816b12b`
- Remote root files resolved: 100, about 1.85 TiB total
- Logical region size for this test: 16 MiB
- Absolute work-item cap: 6
- Logical source bytes processed: 100,663,296 (96 MiB)
- Saved plan: 118,544 total regions
- Work-plan identity hash:
  `dd2c5c7f67879b8bfd4dfd727444038f2cc602f9d19f7d144df98077680c20d9`

The first six shuffled regions came from six different files:

```text
part_93.tokenized.jsonl
part_78.tokenized.jsonl
part_28.tokenized.jsonl
part_69.tokenized.jsonl
part_82.tokenized.jsonl
part_21.tokenized.jsonl
```

## Commands

The uninterrupted reference used:

```bash
uv run python -m dataset.main build \
  --target-tokens 100000000 \
  --max-work-items 6 \
  --region-bytes 16777216 \
  --writer-buffer-bytes 1048576 \
  --checkpoint-bytes-threshold 524288 \
  --output-dir /tmp/small-llm-live-smoke.jKmFCa/reference
```

The interrupted copy used the same settings plus:

```bash
--simulate-crash-after-written-bytes 2500000 \
--output-dir /tmp/small-llm-live-smoke.jKmFCa/resumed
```

It stopped at 2,501,872 bytes on disk. The durable checkpoint confirmed
2,116,336 bytes, leaving a 385,536-byte tail that resume had to truncate.

Resume used:

```bash
uv run python -m dataset.main build \
  --resume \
  --target-tokens 100000000 \
  --max-work-items 6 \
  --region-bytes 16777216 \
  --writer-buffer-bytes 1048576 \
  --checkpoint-bytes-threshold 524288 \
  --output-dir /tmp/small-llm-live-smoke.jKmFCa/resumed
```

Both outputs were checked with:

```bash
uv run python -m dataset.main verify \
  --full-scan \
  --output-dir <reference-or-resumed-directory>
```

## Result

Both runs stopped cleanly at the work-item cap, so `complete` is correctly
`false`; this is a bounded corpus, not a pretend-finished production corpus.

- Documents inspected / accepted: 29,241 / 29,241
- Accepted source tokens: 17,983,618
- Inserted EOD tokens: 29,241
- Total written tokens: 18,012,859
- Train: 29,215 documents, 17,967,192 source tokens, 17,996,407 written tokens
- Validation: 26 documents, 16,426 source tokens, 16,452 written tokens
- Structural rejections: 0
- Cluster 11 accepted documents/tokens: 0 / 0

Observed accepted clusters:

| Cluster | Documents | Source tokens |
|---:|---:|---:|
| 6 | 5,454 | 2,979,959 |
| 7 | 3,062 | 3,013,306 |
| 12 | 10,845 | 6,009,408 |
| 16 | 5,910 | 2,984,821 |
| 17 | 3,970 | 2,996,124 |

Full token-range verification passed for both outputs with no problems.

The uninterrupted and resumed files were byte-identical:

```text
train.bin       89b91d61d7be16a6ef8a4581bf0f45aa48e5a3e8b06a9ff81b020aa55fbfaa50
validation.bin  b0ae30be0f180e94fe712370f50f17db916ba05713f9094c4cfbab3b20a8528b
work_plan.json  4a586239cb13a30d01153cca7b10f311975de88f473912af4bcef24ee88296ad
```

“Source bytes processed” above means owned logical byte regions. Boundary
recovery and HTTP chunking read a little extra around those ranges, by design.
No generated `.bin` file or cache lives in the repository.
