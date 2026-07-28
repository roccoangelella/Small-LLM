# Exact ClimbMix Mixture Calibration

This command recovers the empirical per-cluster source-token distribution of the
complete pinned `nvidia/Nemotron-ClimbMix` release and writes the production
scheduler weights after conditioning on cluster 11 being excluded.

It is a **calibration pass**, not a corpus build. It never writes token shards and
never detokenizes or materializes the large `tokens` arrays. It scans the exact
root `part_*.tokenized.jsonl` files at the immutable revision configured in
`dataset/config.py`, extracts only the top-level `cluster_id` and `token_count`,
and counts every owned JSONL record exactly once.

## Why a complete scan is required

The paper presents rounded mixture percentages. Those values are not precise
enough to reconstruct the released corpus exactly: rounded values need not sum to
100%, and very small clusters may appear as zero after rounding. The release does
not include a separate per-cluster token-total index.

Therefore the only auditable exact weight is:

```text
cluster_total[c] = sum(record.token_count for records with cluster_id == c)
```

The code-free production weight file contains the integer totals for clusters
1-10 and 12-20. Cluster 11 is omitted. The scheduler normalizes these positive
integer totals with exact rational arithmetic, preserving the released corpus's
relative distribution among retained clusters.

## Run

Use a dedicated output directory. The scan reads the complete source release, so
run it on the fast-network production host or another host with comparable
network access.

```bash
uv run python -m dataset.mixture \
  --output-dir /data/climbmix-mixture-calibration \
  --workers 8 \
  --max-in-flight-work-items 16
```

The default 256 MiB work regions and four-work-item checkpoint cadence bound
restart re-download to roughly 1 GiB, plus boundary reconstruction.

Resume with identical source policy:

```bash
uv run python -m dataset.mixture \
  --output-dir /data/climbmix-mixture-calibration \
  --workers 8 \
  --max-in-flight-work-items 16 \
  --resume
```

Worker count and in-flight concurrency may change on resume because they do not
change record ownership or the output. Dataset identity, revision, source glob,
work plan, and completed work-item prefix are validated before continuing.

## Outputs

The output directory contains:

- `work_plan.json`: pinned source files, sizes, regions, order, and self-hash.
- `mixture_progress.json`: crash-safe cumulative counters and next work item.
- `mixture_report.json`: all 20 cluster token/document totals, source coverage,
  conditioning rule, weights hash, and self-hash.
- `climbmix_code_free_weights.json`: the production `--weights-file`, containing
  exact integer token totals for accepted clusters only.

No rounded or hand-written fallback weight file is committed to the repository.

## Approval

After completion:

```bash
sha256sum \
  /data/climbmix-mixture-calibration/climbmix_code_free_weights.json \
  /data/climbmix-mixture-calibration/mixture_report.json

uv run python -m dataset.main stream-cache \
  --weights-file /data/climbmix-mixture-calibration/climbmix_code_free_weights.json \
  --show-stream-config
```

Approve and archive both hashes together with the pinned revision and work-plan
hash. Copy the weight file to the secure production configuration location only
after reviewing:

- `complete` is true;
- `source_bytes_scanned` equals the sum of every pinned source-file size;
- every cluster 1-20 has a positive token count;
- cluster 11 is present in the all-cluster report but absent from the weight file;
- the accepted-cluster token totals in the report exactly match the weight file;
- a completed `--resume` returns the same report and weights hashes.

## Scheduling interpretation

The weight values are **global token proportions**, not a requirement that each
GPU microbatch contain all clusters. The whole-document largest-deficit scheduler
tracks cumulative emitted source tokens and converges toward the calibrated
mixture over its rolling token windows. A one-sequence microbatch may naturally
contain only one cluster.

## Relationship to the 90B run

Do not start the standalone 90B cache build after calibration. First run the
authenticated bounded dataset pilot, finish trainer consumption and joint
checkpoint integration, and run an end-to-end training pilot. At production
launch, start the dataset producer with a bounded cache head start and then start
the trainer so source reading, packing, Drive mirroring, and training overlap.
