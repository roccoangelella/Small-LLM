# Project Status

_Last updated: 2026-08-04_

## Current phase

The fixed finite approximately-20M engineering qualification dataset is built,
durably mirrored, fully scanned, and accepted. Exact-commit Kaggle T4 evidence
and the 20-successful-update constant-LR trainer preflight have now passed.

The project is in **post-preflight repeatability and recovery qualification**.
Architecture selection is not being reopened. The complete 306-update one-pass
segment, approximately-100M architecture comparison, and complete 90B dataset
build remain unauthorized until the remaining qualification ladder passes.

Detailed preflight evidence and measurements are recorded in
`llm_docs/20m_kaggle_preflight_results.md`.

## Fixed model and optimizer

```text
parameters: 20,637,592
architecture: [GDN-2, GDN-2, GDN-2, full gated MHA] repeated
context length: 2,048
precision: FP16
GDN-2 backend: ordinary PyTorch chunkwise
GDN-2 chunk size: 32
initialization: normal
seed: 17
```

Primary optimizer:

```text
ordinary feature-transform matrices: whole-matrix Muon
embedding, norms, biases, dynamics, structured filters: AdamW
base LR: 3e-4
AdamW betas: 0.9 / 0.95
AdamW epsilon: 1e-8
AdamW weight decay: 0.1
Muon momentum: 0.95
Muon LR multiplier: 1.0
Muon target direction RMS: 0.18
Muon weight decay: 0.1
global gradient clipping: 1.0
```

Pure AdamW remains the later matched control. No optimizer or clipping change is
authorized from the short preflight alone.

## Accepted qualification dataset

```text
run ID: 20m-qualification-dataset-001
accepted source tokens: 10,000,662
train source tokens: 9,991,872
validation source tokens: 8,790
train shards: 6
validation shards: 1
train sequences: 4,886
validation sequences: 5
train blocks / one-pass optimizer updates: 306
stored uint16 tokens: 10,021,659
manifest SHA-256: 1e5ee8f372b77b6728288610dbe7cce74d833be21e53d1538bc5a890229b18bb
Drive manifest SHA-256: fbb29ee0d0102658e1274e39d6647cf56a6dcb685e0f566b1736847dcc4fbe84
```

Accepted evidence includes schema-v2 structural and per-shard SHA-256
verification, exact local-to-Drive identity, literal token-by-token scanning of
all stored tokens, no vocabulary or geometry problems, exact cluster accounting,
and exact plan regeneration from the private Kaggle mount.

The dataset is engineering qualification data, not strong model-quality or broad
mixture-coverage evidence.

## Exact one-pass plan

```text
schedule: WSD
passes: 1
steps: 306
full-block target tokens: 32,768
warmup: 16 updates / 524,288 target tokens
stable: 228 updates / 7,471,104 target tokens
decay: 62 updates / 2,011,136 target tokens
minimum LR ratio: 0.1
validation blocks: 1
train target tokens: 10,006,528
```

The final training block is partial. Silent data wraparound remains forbidden.

## Passed exact-commit Kaggle gates

The repository-native launcher ran in an isolated clean detached worktree at:

```text
launch commit: 45d1da4a1ac3f18cf6ce02b8439672f10e2c8b4c
GPU: Tesla T4
Python: 3.13.14
PyTorch: 2.13.0+cu130
CUDA runtime: 13.0
```

Results:

```text
offline suite: 229 passed, 1 expected live-remote skip
corrected T4 harness: passed
dataset full scan: passed
exact qualification plan reproduction: passed
20-update trainer preflight: passed, exit code 0
summary status: passed_preflight
authorization: post_preflight_review_only
```

The corrected T4 harness selected the frozen primary candidate:

```text
architecture: gdn2_hybrid
backend: pytorch_chunkwise
chunk size: 32
precision: FP16
throughput: 1,125.52 target tokens/s
overflow count: 0
peak allocated: 2,346.19 MiB
peak reserved: 2,456 MiB
```

## Twenty-update preflight findings

The trainer consumed exactly 20 blocks / 655,360 target tokens at constant
`3e-4` and produced checkpoint `step-00000020`.

```text
training loss: 10.845867 -> 9.573909
validation loss: 9.240405 on 10,240 target tokens
mean throughput: 1,066.12 target tokens/s
throughput range: 1,026.47 to 1,075.82 target tokens/s
maximum allocated CUDA memory: 2,393.83 MiB
maximum reserved CUDA memory: 2,868 MiB
GradScaler: stable at 65,536
overflow events / retries: 0 / 0
checkpoint byte size: 216,852,669
checkpoint save time: 2.094 s
```

Muon direction RMS remained approximately `0.18`; its effective
update-to-weight ratio remained approximately `0.003094`. AdamW branch update
statistics remained finite and decreased smoothly.

### Optimizer-stability review flag

Gradient clipping occurred on all 20 successful updates. Pre-clip global norms
ranged from `1.3763` to `2.7359`, with the maximum at update 20. Loss decreased
and no non-finite behavior occurred, so this is not a hard correctness failure.
It does, however, exceed the protocol's provisional clipping-frequency bands.

The longer reference and A/A runs must establish whether the gradient-norm rise
is repeatable and bounded. Do not silently change LR or clipping norm. A
one-variable diagnostic is required before any recipe replacement if clipping
remains nearly universal or norms keep growing.

## Evidence identity

```text
evidence directory: /kaggle/working/small-llm-qualification-controller/small-llm-qualification-20260804T135359Z
summary: /kaggle/working/small_llm_qualification_summary.json
W&B run ID: 20m-t4-preflight-001
offline log SHA-256: b5889c8476039b4a99717e0bca58502095d980db8b1b00584ccbec8f095fad47
T4 log SHA-256: 5bd020caebf393fef0a0b4cec87f8c52f0428eb821b3dc2e63fccdd1765c9569
preflight log SHA-256: e23cfd896dbc8f53613b8b42e1fd9b1ec4d337f64753bb1b857e5f64c5d35e90
```

## Remaining qualification sequence

1. Preserve the full Kaggle evidence directory and W&B run.
2. Run an uninterrupted reference segment of at least 50 successful updates
   from a known initial state using the precisely matched WSD prefix.
3. Run a second uninterrupted same-hardware A/A segment from the same state,
   seed, data order, and recipe.
4. Quantify the nondeterministic floor and freeze empirical warning/failure
   thresholds for loss, throughput, memory, overflow, clipping, gradient norms,
   and optimizer update statistics.
5. If the clipping pattern remains nearly universal or grows without bounding,
   stop for a separately labeled one-variable diagnostic; do not silently alter
   the frozen recipe.
6. Qualify actual-process local interruption at the planned update-25 checkpoint
   boundary and exact resume against the uninterrupted reference.
7. Qualify private remote publication and empty-environment restore, including
   verified two-shard prefetch and exact next-block continuation.
8. Authorize and run the complete 306-update one-pass segment only after all
   preceding gates pass.
9. Run final validation and deterministic generation, then record the final
   qualification report before considering the approximately-100M comparison.

## Current readiness verdict

**Dataset gate: passed.**

**Exact-commit offline/T4/mounted-data gates: passed.**

**Twenty-update integrated trainer preflight: passed for execution and
integration.**

**Ready next:** uninterrupted 50-update reference and same-hardware A/A
repeatability runs, followed by threshold freeze and recovery qualification.

**Not yet authorized:** the complete 306-update one-pass segment, because
repeatability, clipping interpretation, local interruption/resume, and remote
empty-environment recovery remain outstanding.
