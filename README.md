# Small-LLM dataset pipeline

`dataset/` contains a resumable curation pipeline for an approximately 400 GB,
90B-token natural-language subset of NVIDIA Nemotron-ClimbMix.  It uses the
official GPT-2-tokenized source, the source cluster IDs as the primary signal,
and deterministic code/API-dump exclusion as a guardrail.

All tunable values, including the 20 cluster policies and per-cluster quota
percentages, live in [dataset/config.py](dataset/config.py).  The default policy
uses NVIDIA's published numeric cluster map, which we checked on bounded live
samples in [cluster_map_validation.json](cluster_map_validation.json). No numeric
cluster is thrown away wholesale: code-heavy clusters 1, 6, 11, 12, and 18 keep
useful prose but lose source code, repositories, and generated API material.
Cluster 20 is civic/political material, not the Python cluster.

Install dependencies once:

```bash
uv sync
```

Run the stages explicitly:

```bash
# Streams the official source once. Writes 50 deterministic samples per cluster,
# eligibility inventory, and a compact manual spot-check worksheet.
uv run python -m dataset.main sample

# Sends review batches to the local GemRouter model gemini-3.6-flash.
uv run python -m dataset.main review

# Inspect artifacts/manual worksheet, edit config.py only if warranted, then:
uv run python -m dataset.main plan

# Streams again and writes JSONL shards; checkpoints make interruption safe.
uv run python -m dataset.main select
uv run python -m dataset.main select --resume

# Fresh deterministic + Gemini audit of the final output.
uv run python -m dataset.main audit
```

The official source contains GPT-2 token IDs rather than raw text, so text is
decoded before sampling, code detection, review, writing, and auditing.  The
pipeline deliberately does not begin the long production streams automatically.
