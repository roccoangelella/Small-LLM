# 20M Model / 100M-Token Data-Scaling Experiment

_Last updated: 2026-08-05 15:18 Europe/Rome_

## Decision

The next authorized experiment keeps the completed approximately-20M model family and increases the finite training dataset from approximately 10M to approximately 100M accepted source tokens.

The current experiment does **not** authorize a 1B-, 10B-, or 100B-token run and does not enlarge the model. Future dataset-size experiments should normally use approximately logarithmic boundaries such as 10M, 100M, 1B, 10B, and the final approximately-90B production scale, but every later boundary remains a separate decision.

## Scientific variable

The model, tokenizer, source corpus identity, source revision, accepted/excluded cluster policy, mixture weights, initialization family, seed, architecture, optimizer, context length, sequences per optimizer block, precision, and one-pass policy remain unchanged from the accepted 20M/10M qualification.

The intended scientific change is:

```text
accepted-source-token target: 10,000,000 -> 100,000,000
minimum accepted source tokens: 9,000,000 -> 90,000,000
maximum accepted source tokens: 11,000,000 -> 110,000,000
```

The model remains:

```text
parameters: 20,637,592
model size: smoke
architecture: gdn2_hybrid
pattern: [GDN-2, GDN-2, GDN-2, full gated MHA] x 2
context length: 2,048
GDN-2 chunk size: 32
initialization: normal
precision: FP16
seed: 17
```

The optimizer remains:

```text
hybrid whole-matrix Muon + AdamW
base LR: 3e-4
AdamW betas: 0.9 / 0.95
AdamW epsilon: 1e-8
weight decay: 0.1
Muon momentum: 0.95
Muon LR multiplier: 1.0
Muon target direction RMS: 0.18
Muon weight decay: 0.1
global gradient clipping norm: 1.0
WSD warmup/stable/cosine-decay policy
minimum LR ratio: 0.1
```

## Dataset and shard policy

The 100M finite dataset remains an immutable schema-v2 shard set. It is not converted into one monolithic training file and it is not streamed from Google Drive or Hugging Face during optimizer steps.

```text
run ID: 20m-100m-dataset-001
context length: 2,048
stored tokens per sequence: 2,049
sequences per optimizer block: 16
target tokens per full optimizer update: 32,768
target shard size: 8 MiB
source-production durable checkpoint cadence: 20,000,000 source tokens
remote durability: required
passes: 1
implicit wraparound: forbidden
```

The source-production durable checkpoint cadence is scaled by 10x from the 10M build so the 100M producer retains approximately the same number of durable production boundaries. The immutable shard size remains 8 MiB.

For Kaggle training, the complete approximately-200-MB uint16 dataset is attached and mounted once under `/kaggle/input`. The trainer reads local immutable shards sequentially. Google Drive remains the durable mirror and recovery source, but ordinary optimizer steps perform no network shard downloads.

The single producer source of truth is:

```text
dataset/qualification_100m.py
dataset/qualification_100m_report.py
```

Superseded duplicate 100M profile modules were removed.

## Increased microbatch execution

The effective optimizer batch is unchanged at one 16-sequence prepared block, or 32,768 target tokens per full update.

The candidate execution grouping is:

```text
microbatch size: 4
microbatches per full block: 4
optimizer updates per block: 1
```

This changes only how the same block is split into forward/backward calls. It does not change the number of sequences or target tokens contributing to an optimizer update.

Because FP16 accumulation order and GPU kernels can differ with microbatch size, microbatch 4 is not silently assumed equivalent. On the first session, the launcher runs microbatch 1 and microbatch 4 from the same seed, initialization, schedule, and first eight blocks. Microbatch 4 is accepted only when:

- block IDs, target-token counts, consumed-token cursors, and learning rates match;
- all losses, gradients, scaler values, and throughput values are finite;
- no FP16 overflow or retry occurs;
- median throughput is at least 5% higher after discarding two warm-up updates;
- maximum per-step loss difference is at most 0.05;
- maximum relative gradient-norm difference is at most 5%;
- peak reserved memory is at most 90% of the T4's physical memory.

If the gate fails, the launcher fails closed. It does not silently fall back, alter the effective optimizer batch, or change another hyperparameter.

## Schedule and persistence cadence

The exact update count and token horizons are derived from the completed, fully verified 100M manifest. The same fractional WSD policy is retained:

```text
warmup updates: max(16, ceil(5% of planned updates))
decay updates: ceil(20% of planned updates)
stable updates: all remaining updates
minimum LR ratio: 0.1
```

Operational cadence is scaled by approximately 10x in optimizer steps:

```text
local checkpoint: every 250 successful updates
validation: every 500 successful updates
remote checkpoint publication: every 500 successful updates
```

A final validation, local checkpoint, and verified remote publication remain mandatory at the actual end of every bounded Kaggle segment and at the final update.

## Bounded Kaggle segments and exact resume

A 100M-token run can exceed one Kaggle notebook session. The launcher therefore does not rely on an abrupt notebook timeout.

```text
maximum additional updates per session: 749
cross-session authority: private Hugging Face latest pointer
attached data source on every session: the same immutable Kaggle shard dataset
W&B run ID: 20m-100m-data-001
```

Each invocation:

1. verifies the complete attached dataset and regenerates the same exact one-pass plan;
2. checks for the private remote `latest.json` pointer for run `20m-100m-dataset-001`;
3. starts fresh only when that pointer is absent;
4. otherwise downloads and verifies the published checkpoint tree;
5. requires the checkpoint's embedded Drive manifest to hash-identically match the attached dataset's Drive manifest;
6. requires checkpoint step `N` to correspond to `last_consumed_block_id = N - 1`;
7. restores model, optimizer, scheduler, scaler, RNG, and dataset cursor;
8. resumes the same W&B run with `wandb-resume=must`;
9. executes at most 749 additional updates;
10. exits normally only after a final verified remote publication.

The segment planner avoids ending a non-final segment exactly on a 500-step periodic-publication boundary so the trainer emits an explicit `final=true` remote-publication event.

Segmenting is an operational accommodation, not a scientific variable. Exact local and remote interruption/resume were already qualified on the 20M recipe, and the effective optimizer update remains one complete 16-sequence block.

## Single Kaggle entry point

The official operator command is:

```text
%cd /kaggle/working/Small-LLM
!git pull --ff-only
!python kaggle/run_20m_100m.py
```

Run the same command again after each successful bounded segment. The entry point pins the implementation commit and rejects a caller-supplied launch-commit override.

Required Kaggle configuration remains:

```text
accelerator: NVIDIA T4
internet: enabled
attached input: completed 20m-100m-dataset-001 shard dataset
secrets: WANDB_API_KEY, HF_TOKEN, SMALL_LLM_HF_REPO_ID
optional secret: WANDB_ENTITY
```

The implementation records full-scan evidence, the exact plan, microbatch gate results, restore evidence, commands, logs, exit codes, hashes, segment boundaries, and the final summary under `/kaggle/working`.

## Test coverage

Offline tests cover:

- the exact 100M producer envelope and 20M-source-token producer checkpoint cadence;
- exact WSD derivation and partial-final-block handling;
- Drive-manifest identity binding;
- segment planning and avoidance of periodic-publication endpoints;
- resume CLI arguments and stable W&B identity;
- microbatch acceptance and rejection thresholds;
- dataset-profile rejection when the producer cadence is wrong;
- explicit final remote-publication evidence.

The pure launcher tests were additionally executed locally with all five tested behaviors passing. Full T4 and live-remote behavior remains a launch-time qualification because those surfaces require Kaggle, W&B, Hugging Face, and the completed attached dataset.

## Interpretation boundary

This 100M-token experiment is a data-scaling experiment for the already-qualified 20M model. It is intended to measure how much the 10M checkpoint's weak factual retrieval, topic drift, repetition, and short-range coherence improve when the same model receives approximately 10x more distinct training data.

It is not the approximately-100M-parameter architecture comparison. Model enlargement remains a later, separately authorized stage.