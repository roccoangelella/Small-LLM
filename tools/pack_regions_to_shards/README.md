# pack_regions_to_shards — from the 100B v2 region corpus to schema-v2 trainer shards

Reads the region files of `roccoangelella/small-llm-corpus-100b-v2-workers` in work-plan order,
drops cross-region exact duplicates (same rule as the builder's merge: document `id` already seen),
and drives this repository's own producer classes — `SequencePacker` semantics, `PreparedBlock`,
`ImmutableShardWriter`, `TokenDeficitScheduler`, `is_validation`, `verify`, `build_run_contract`,
`publish_frontier`, `HuggingFaceBucketShardStore` — so the output is the same schema-v2
`context_plus_one` format the trainer consumed for run 001, verified by `dataset.src.verify.verify`.

`fast_packer.FastBlockPacker` is a vectorised replacement for the per-token `SequencePacker` loop
(4.7 s per 50M-token region instead of 54 s). It is validated byte-for-byte against the original on
real regions: `--packer slow` runs the reference path for the comparison.

Two deliberate deviations from `dataset.production`, both recorded in the manifest:
- documents are emitted in corpus order; the deficit scheduler's re-stratification is bypassed
  (`emit()` is still called, so the manifest's scheduler block is truthful);
- the tokenizer contract names the file the regions were built with (`superbpe_8000_v2`, sha256
  068e20ef…), not `superbpe_8000.json` (run 001's, a different vocabulary under the same name).

`config.EOD_TOKEN_ID` is forced to 7992 at import: the producer classes insert it, and its default
is GPT-2's 50256, which is outside the 8000-entry vocabulary.

    python tools/pack_regions_to_shards/pack_shards.py --regions 2100 \
        --run-id moe-100b-superbpe-b64-dataset-002 --out /path/window \
        --bucket roccoangelella/small-llm-corpus-100b-v2-dataset --prefetch 16 --checkpoint-regions 10

Resumable: a crash after any checkpoint resumes from `pack-state.json` + `seen.bin` and reproduces
the uninterrupted output byte for byte (tested). Local disk stays a window of a few GB.
