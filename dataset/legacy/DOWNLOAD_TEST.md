# Legacy download test

This is the old decoded-text/filter experiment. Its conclusions about running a
filter and balancing cluster quotas are superseded by the production policy in
`../README.md`. It stays here only as a record of the early investigation.

The point of this test was simple: can we stream the official Nemotron-ClimbMix shards and run the real decode and filter path without downloading the whole dataset?

## What we actually ran

There were three useful passes. First, opencode walked a contiguous 180-second prefix through the source iterator. It processed 55,766 documents and 50.2M GPT-2 tokens at about 309 documents per second. The prefix contained only cluster 1. That did not mean the whole dataset was cluster 1; it showed that the beginning of the stream is one large region and is not a representative sample. The same speed gives a rough full-stream estimate of about 399 hours for 400B tokens, so reading everything just to understand the layout would be painful.

Second, we made 120 deterministic random reads across the 100 root shards. Each read sought to a random byte offset and read a 600 KB window, not a full shard. We discarded the partial first and last JSONL records, decoded the GPT-2 token IDs, and ran the complete production hash and text-filter path on the remaining documents. This checked 5,952 documents (4.24M source tokens) from 72 MB of data. There were no read failures; 5,901 documents passed and 51 were rejected, mostly because they were too short.

The random pass saw 16 of the 20 clusters, so clusters 2, 5, 14, and 19 looked missing. The follow-up placed 200 tiny 8 KB probes roughly every 10 GB across the source, using the cluster ID in the next complete record to map the large byte regions. It then read targeted 600 KB windows in the missing regions. A finer 80-probe scan was needed around cluster 14. Altogether this added 350 documents from clusters 2, 5, 14, and 19, and all 350 passed the deterministic filter.

We also settled which cluster map to use. The numeric IDs follow NVIDIA's published CLIMB topic table, not the earlier project map or the conflicting Hugging Face interpretation. A bounded check of 100 documents found 11 direct matches, 6 partial matches, and 3 broad-topic mismatches. That is good enough as the working map, but these are broad semantic buckets rather than perfectly pure topics, so the larger review and manual check still matter.

So the streaming approach works, but random byte sampling alone is not enough. The production pass needs cluster-aware sampling so that every configured cluster gets its share. This was only a download and coverage test; the real target is still about 90B tokens, streamed into the planned output rather than loaded into the LLM at once.
