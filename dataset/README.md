The dataset is huge, so we do not download the whole thing and then wonder where the disk went. ClimbMix is streamed from huggingface: one document comes in as GPT-2 token ids, tiktoken brings it back to normal text, we decide if it is useful, and only then we save it. Therefore the large thing on disk is our final dataset, not the whole 400B-token NVIDIA dataset plus our dataset on top of it.

The first pass is basically the “what is even inside this thing?” pass. We take 50 deterministic documents from every `cluster_id`, so it is not just the first 50 docs NVIDIA happened to put in a shard. Gemini reads them in small batches, then there is a smaller manual worksheet to see if Gemini is saying something silly.

The numeric topic map in `config.py` is [NVIDIA's published CLIMB map](https://research.nvidia.com/labs/lpr/climb/), not a set of guesses from the cluster numbers. We checked it against 100 bounded live samples in [`../cluster_map_validation.json`](../cluster_map_validation.json): 11 direct matches, 6 partial matches, and 3 broad-edge mismatches. Clusters are semantic buckets, not single-subject folders. The practical consequence is that cluster 11 is programming, 15 is film/comics, 16 is climate, 18 is security/networking, and 20 is civic/political material. We do not exclude a whole numeric cluster for code. Instead, clusters likely to contain it (1, 6, 11, 12, and 18) are explicitly marked `keep_without_code`, while the deterministic code filter applies to every selected document anyway.

Gemini does not get to just say “yes, cluster 8 sure sounds like gaming” and call it a day. Every batch has to return the same JSON object: observed topics, `topic_alignment` (`match`, `partial_match`, `mismatch`, or `unclear`), short evidence, language and code estimates, a keep/exclude recommendation, and confidence. The expected topic and current policy are shown as hypotheses, but the prompt explicitly says to form the topic judgement from the samples first. Malformed or incomplete JSON is rejected and retried before it is written. This gives us comparable results across all clusters, while still letting the model flag a broad or poor-quality pocket inside an otherwise sensible semantic cluster.

While the samples are being made we also count how much usable text every cluster has. This is useful because we do not want to randomly select 90B tokens and discover at the end that it is 80% medicine or 80% history. The plan turns the per-cluster percentages in `config.py` into deterministic hash sampling rates. A second stream then writes only the selected text, stopping around 90B tokens / 400gb. It does not need to keep the original shards locally.

There is still a code filter even for the “good” clusters. It removes source files, repository-like dumps, large fenced-code blocks, and generated API-reference-shaped pages. Technical prose is fine: a document explaining networking, operating systems, physics, or algorithms is exactly the kind of thing we want. The final audit takes a new sample from the output, reruns the checks, and asks Gemini again. So hopefully we do not accidentally train the little model to autocomplete Python tutorials all day.

Run the stages in order:

```bash
uv run python -m dataset.main sample
uv run python -m dataset.main review
uv run python -m dataset.main plan
uv run python -m dataset.main select
uv run python -m dataset.main audit
```

All the numbers and cluster decisions live in `config.py`. The pipeline code is split in `src/`, because one giant “do everything” file becomes annoying very fast.
