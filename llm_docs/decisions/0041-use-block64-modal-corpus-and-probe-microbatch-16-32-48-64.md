---
status: accepted
date: 2026-08-11
supersedes: 0040
---

# Use a block-64 Modal corpus and probe microbatch 16/32/48/64

## Decision

For the authorized approximately-100M-parameter / 2B-token Modal pretraining run, replace the earlier 16-sequence optimizer-block plan from ADR 0040 with a 64-sequence prepared-block dataset derived byte-for-byte from the verified `20m-2b-dataset-001` corpus.

The Modal run contract is now:

- model preset: `100M` / trainer `substantive`;
- token budget: the exact existing 2B finite corpus, preserving stored context+1 sequence bytes and split order;
- Modal dataset profile: `modal-2b-b64`;
- Modal dataset run ID: `modal-2b-b64-dataset-001`;
- context length: 2,048;
- optimizer batch: one 64-sequence prepared block, approximately 131,072 target tokens for each full optimizer update;
- execution microbatch qualification candidates: 16, 32, 48, and 64;
- qualification selection: fastest candidate that completes the real forward/backward probe with finite loss/gradients and no more than 90% reserved GPU memory;
- GPU request: `H100`, retaining the compatible Modal H200 upgrade path;
- precision: FP16 autocast with FP32 master parameters;
- optimizer, GDN-2 execution, initialization, seed, and token-based WSD settings remain unchanged;
- checkpoint cadence: every 250 successful optimizer updates plus the final checkpoint, durably stored in `small-llm-runs`;
- validation cadence: every 250 successful optimizer updates;
- W&B: online `Small-LLM`, stable run ID `100m-2b-data-001`, exact resume semantics.

The existing 16-sequence 2B corpus remains immutable and canonical for the active/historical Kaggle 20M trajectory. It is not edited in place.

## Dataset derivation

The new dataset is a physical reblocking only. `dataset/reblock.py` verifies the existing 2B source corpus, copies its train and validation shard byte streams in exact order, groups those same fixed records into 64-sequence prepared blocks, emits new shard checksums/manifests, and verifies the derived qualification plan. No source documents are downloaded and no tokenization, packing, split assignment, mixture scheduling, or record content is recomputed.

The profile uses a 32 MiB target shard size. This aligns naturally with the legacy 8 MiB / 16-sequence geometry: four ordinary full legacy shards contain the same number of sequence bytes as one ordinary block-64 shard grouping.

The existing 2B training stream contains 976,560 sequences. Reblocking therefore changes the one-pass optimizer-update count from 61,035 at block 16 to 15,259 at block 64, with the final block containing 48 sequences. The warmup/stable/decay boundaries remain identical in token space under the manifest-derived WSD plan:

```text
warmup tokens: 100,007,936
stable tokens: 1,499,987,968
decay tokens: 399,998,976
total train target tokens: 1,999,994,880
```

## Rationale

The H100/H200 should be allowed to expose substantially more execution parallelism than the T4-era 16-sequence block permitted. Keeping the storage/optimizer block at 16 would make microbatch sizes above 16 impossible to measure and could leave expensive Hopper capacity unused.

A 64-sequence block is large enough to benchmark 16, 32, 48, and 64 without introducing cross-block transactional accumulation into the trainer. The hardware probe remains authoritative: 48 or 64 may fail on an 80 GB H100 with the current activation-heavy attention/LM-head implementation, while an H200 may admit a larger candidate. A failed candidate is diagnostic and does not alter the production trajectory; the fastest safe candidate is frozen before optimizer step 1.

The optimizer batch itself is intentionally changed by this decision. This is a new 100M / 2B trajectory, so the efficiency gain is preferred over preserving the earlier T4-oriented 32k-token update size. Token-based schedule boundaries are retained so the data exposure and WSD phase budget remain directly interpretable.
