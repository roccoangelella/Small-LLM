# Nemotron-ClimbMix production builder

There is one production path in this folder now. It reads the source token IDs,
checks their structure and `cluster_id`, and appends them to two binary files.
It never turns normal accepted documents back into text.

The old sample → Gemini review → plan → JSONL select → audit workflow is parked
under `legacy/` for history. It is not exposed by the CLI and its old optional
dependencies are gone.

## Commands

Install the locked environment:

```bash
uv sync --locked
```

Start the real build, resume it, or verify a finished corpus:

```bash
uv run python -m dataset.main build
uv run python -m dataset.main build --resume
uv run python -m dataset.main verify
```

The default build is the 90B-token production job. A bounded connectivity test
uses the same code with smaller limits:

```bash
uv run python -m dataset.main build \
  --target-tokens 10000000 \
  --max-work-items 20 \
  --output-dir /tmp/climbmix-smoke

uv run python -m dataset.main verify \
  --output-dir /tmp/climbmix-smoke \
  --full-scan
```

`--max-work-items` is an absolute cap in the saved shuffled plan. If a bounded
run is interrupted after item 7 and resumed with `--max-work-items 20`, it stops
after item 20 in total, not after 20 more items. This is what makes bounded
interrupted and uninterrupted runs directly comparable.

## Frozen production policy

- Source: `nvidia/Nemotron-ClimbMix`
- Revision: `5eaa64b9c0c85b7f56af01d7dffdb0795816b12b`
- Files: root `part_*.tokenized.jsonl` only; `climbmix_small` and every
  subdirectory are excluded
- Target: 90,000,000,000 accepted source tokens
- Minimum acceptable completed size: 80,000,000,000
- Hard maximum: 100,000,000,000
- Accepted clusters: 1–10 and 12–20
- Excluded cluster: 11, NVIDIA's software/programming cluster
- Seed: `small-llm-climbmix-production-v1`
- Validation probability: 0.001 per document
- Tokenizer: the source GPT-2 token IDs, reused as-is
- End-of-document token: GPT-2 `<|endoftext|>`, ID 50256

There are no per-cluster quotas. Every root shard is divided into equal 256 MiB
logical regions (apart from each file's final short region), and all regions
are deterministically hash-shuffled. Processing that order preserves the source
mixture approximately without forcing every cluster to have the same size.

Every work item stores the full source revision, filename, range start, and range
end in its stable identity. Every record is identified by revision, filename,
and its absolute JSONL record-start byte. Adjacent ranges can fetch overlapping
boundary bytes, but a record belongs only to the half-open range containing its
first byte. That is what prevents duplicates and dropped lines.

## What gets validated

Production validation is structural only:

- the JSON record parses;
- `cluster_id` is an integer from 1 through 20;
- `tokens` is a non-empty list;
- every token is an integer from 0 through 50256;
- `token_count`, when present, equals `len(tokens)`.

Invalid records are counted by reason, logged with revision/file/byte offset,
and skipped. `--strict` changes that last behavior and aborts on the first
invalid record. Cluster 11 is structurally valid but rejected by the numeric
cluster policy.

There is no detokenization, retokenization, language filter, code-density
filter, text-quality filter, semantic classifier, or LLM call in this path.

## Output format

The default output is:

```text
dataset/output/
├── train.bin
├── validation.bin
├── progress.json
├── work_plan.json
└── manifest.json
```

`train.bin` and `validation.bin` are raw little-endian unsigned 16-bit token
IDs. They have no header, JSON framing, compression, or native-endian fields.
ID 50256 is appended after a document only if that document does not already
end in 50256.

`accepted_source_tokens` counts token IDs already present in accepted source
documents. `written_tokens` counts those source tokens plus inserted EODs. The
90B stop condition uses `accepted_source_tokens`.

The train/validation choice is a versioned SHA-256 hash of the fixed seed,
source revision, filename, and absolute record-start byte. It does not depend on
processing order or machine state, and one document can only land in one file.
The default writer keeps up to 256 MiB per active output in memory and flushes
large blocks.

The entire `dataset/output/` directory is ignored by Git.

## Checkpoints and resume

The default checkpoint threshold is 1 GiB of newly written binary data. A
checkpoint is committed in this order:

1. write both in-memory buffers;
2. flush both files;
3. `fsync` both files;
4. atomically replace and directory-`fsync` `progress.json`.

The checkpoint records confirmed byte sizes, the work-item/record cursor, split
and cluster counters, inspected records, structural rejections, the work-plan
hash, and a configuration hash.

On `--resume`, the builder validates every output-defining setting and refuses
changed source, plan, seed, split rule, cluster policy, binary format, target,
strictness, or bounded work-item cap. It then truncates both binaries to the
confirmed sizes before appending. Bytes written after the last checkpoint are
supposed to disappear; the same records are read again from the saved cursor.

Changing writer-buffer or checkpoint sizes is safe because those settings
affect throughput, not output identity.

## Disk and network preflight

The builder checks that the output directory is writable, the filesystem can
represent the projected large file, enough space is available, the pinned
source tree still has the expected root files/sizes, and several raw source
ranges are reachable.

The conservative free-space formula is:

```text
target source tokens × 2 bytes × 1.02 EOD allowance × 1.30 safety margin
```

For 90B tokens this is about 238.7 GB (222.3 GiB). Resume subtracts already
confirmed output bytes from that requirement. `--allow-unsafe-low-disk` bypasses
the capacity and large-file checks explicitly; it does not change any corpus
policy.

The builder uses direct HTTP range requests. It does not use a sequential
Hugging Face iterator, download whole shards, or put a Hugging Face cache inside
the output corpus.

## Finalization and verification

After the target is reached, both binaries are flushed and fsynced, counters are
checked, SHA-256 is streamed over both binaries and `work_plan.json`, and
`manifest.json` is atomically written. Only then is `progress.json` marked
complete.

Ordinary verification checks the schemas, policy, work plan, sizes, hashes,
checkpoint consistency, and sampled token ranges. Both binaries are read with
`mmap` and explicit little-endian `uint16`; they are never loaded whole into
RAM. `--full-scan` checks every token and is intended for smoke corpora.

Run the local offline suite with:

```bash
uv run python -m unittest discover -v
```

The bounded live result, including interruption/resume hash equality, is in
[PRODUCTION_SMOKE_TEST.md](PRODUCTION_SMOKE_TEST.md).

## What the live test taught us

The short version is that the byte-range approach works where it matters. We read
six shuffled 16 MiB regions from six different source shards, for 96 MiB of source
JSONL in total. That produced 29,241 accepted documents and 17,983,618 accepted
source tokens without walking the dataset from row zero or downloading a complete
shard.

The source is pretty chunky by cluster. Each sampled region was dominated by one
cluster, while the shuffled set as a whole reached clusters 6, 7, 12, 16, and 17.
That is a useful confirmation of why the production plan samples byte regions
uniformly across all files: reading one long sequential stretch would preserve
local ordering more than it would preserve the overall mixture. Cluster 11 did
not happen to occur in these six regions, so this live run did not exercise that
rejection branch. The offline tests cover it directly.

The split and EOD overhead landed close to the assumptions in the configuration.
Validation received 26 of 29,241 documents, about 0.089% against the 0.1% target.
We inserted 29,241 EOD tokens, about 0.16% on top of the accepted source-token
count. The final files contained 18,012,859 tokens and occupied exactly
36,025,718 bytes, which is the expected two bytes per little-endian `uint16`
token. No structurally invalid records appeared in this small sample.

The useful mistake was trying 1 MiB regions first. At this source size that
created roughly 1.9 million work items, making `work_plan.json` unnecessarily
large. We stopped that attempt and used 16 MiB regions for the bounded test,
which reduced the plan to 118,544 items. The 256 MiB production default exists
for the same reason: small regions give tighter test bounds, but they make the
plan and its bookkeeping explode.

Resume was the part worth being fussy about. We interrupted one build after it
had written 385,536 bytes beyond its last durable checkpoint. On resume, the
writer truncated that uncommitted tail, continued from the saved source
position, and produced byte-for-byte identical train and validation files to an
uninterrupted run over the same plan. Both final corpora then passed the normal
verifier and the full token-range scan.

The observed source-reading rate was only about 0.4–0.5 MiB/s. That was a cold,
small, random-range smoke test rather than a production benchmark, so it would be
misleading to turn it into a precise completion estimate. It is still a warning
that the full build will be a long network job and that throughput should be
watched before committing to the 90B-token run.

## License and limitations

Nemotron-ClimbMix is published by NVIDIA under CC BY-NC 4.0. Keep NVIDIA's
attribution and the generated manifest with any derived corpus. The repository
and dataset card are at
<https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix>.

This corpus is cluster-filtered, not guaranteed code-free. Excluding cluster 11
removes the explicit programming bucket, but accepted broad clusters can still
contain incidental code, noisy text, or off-topic documents. That is an
intentional trade-off of using `cluster_id` as the only semantic signal.
