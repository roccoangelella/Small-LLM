# Approximately-20M T4 Qualification Protocol

_Last updated: 2026-08-03_

## Purpose

This document defines how the first approximately-20M-parameter model is qualified on one NVIDIA T4. The goal is engineering evidence, not a model-quality claim. The run must show that the selected model, hybrid Muon + AdamW optimizer, FP16 execution, schema-v2 data path, checkpoint system, interruption/resume behavior, validation path, generation path, and remote recovery path work together without hidden configuration drift.

A successful qualification authorizes the approximately-100M architecture comparison. It does not authorize the complete 90B dataset build by itself, and it does not establish final pretraining hyperparameters for larger models.

## Terminology: finite qualification dataset

Earlier project notes used the phrase **bounded cache**. That wording was technically correct but unclear and should not be used without explanation.

The intended concept is a **finite qualification dataset**:

- the dataset producer reads the pinned source and accepts documents until a configured source-token target is reached;
- it must stop no lower than a configured minimum and never exceed a configured hard maximum except where the existing whole-document boundary contract explicitly determines the final accepted count;
- the accepted text is tokenized and packed into immutable schema-v2 shard files;
- those packed shard files are a cache because the trainer reads the prepared sequences instead of tokenizing the source again;
- the build is finite because it has an explicit token envelope and completion condition rather than growing toward the full 90B production target.

The term does **not** mean:

- context-window length;
- optimizer batch size;
- number of epochs;
- a circular buffer;
- automatic eviction;
- a shortened individual document;
- a data sample that the trainer may silently repeat.

Source-token target and training-token budget are separate quantities. Approximately:

```text
training tokens consumed
= usable training target tokens in the finite dataset
× number of complete or partial passes made by the trainer
```

The first qualification should normally use one pass. Repeating the same finite dataset is an explicit later decision and must not happen implicitly when the trainer reaches the end of the manifest.

The already accepted 10M operational dataset pilot and the new training dataset have different purposes. The former qualified source reading, durability, interruption, resume, verification, and Drive idempotence. Its 512-sequence blocks are retained unchanged as accepted operational evidence. The training dataset must be built separately with 16 sequences per block so one atomic optimizer update contains approximately 32,768 target tokens.

The exact source-token target/minimum/maximum for the new finite qualification dataset remains a separate decision. The previously discussed 10M/9M/11M envelope is a proposal, not silently frozen by this document.

## Fixed model and execution profile

```text
model geometry: approximately 20M parameters
architecture: gdn2_hybrid
context length: 2,048
precision: FP16 model execution
GDN-2 backend: ordinary PyTorch chunkwise
GDN-2 chunk size: 32
initialization: normal candidate
microbatch size: 1
sequences per atomic prepared block: 16
approximate target tokens per optimizer update: 32,768
seed: 17
```

One complete prepared block is one atomic optimizer update. The trainer may split the 16 sequences into microbatches for memory, but it may not acknowledge, checkpoint, or resume in the middle of that block.

## Standard qualification hyperparameter policy

The user selected reasonably standard, conservative hyperparameters instead of a broad hyperparameter search for this engineering run. “Standard” is made concrete below so the run is reproducible and the word cannot conceal a future default change.

### Optimizer split

```text
optimizer: hybrid whole-matrix Muon + AdamW
AdamW beta1: 0.9
AdamW beta2: 0.95
AdamW epsilon: 1e-8
AdamW weight decay: 0.1
Muon Nesterov momentum: 0.95
Muon learning-rate multiplier: 1.0
Muon target update RMS: 0.18
Muon weight decay: 0.1
global gradient clipping norm: 1.0
```

The existing explicit parameter routing remains authoritative. Ordinary feature-transform matrices go to Muon; embeddings, norm scales, biases, GDN dynamics, and structured temporal filters go to AdamW. Unknown trainable parameters fail optimizer construction.

The AdamW epsilon remains the conventional `1e-8` used by this implementation. DeepSeek-V4 reports `1e-20`, but copying that frontier-scale value is not necessary for this small single-GPU qualification and is not treated as a standard requirement.

### Base learning rate

```text
base AdamW learning rate: 3e-4
Muon effective learning rate: 3e-4
```

The Muon branch reuses the base AdamW learning rate through update-RMS normalization and a multiplier of `1.0`.

This value is close to the public DeepSeek-V4-Flash peak learning rate of `2.7e-4` and lies in the conventional range for small decoder-only pretraining with AdamW-style scaling. It is an engineering baseline, not a claim of optimality.

There is no planned broad learning-rate sweep. If the standard baseline fails a hard stability gate, the recovery procedure may run narrow diagnostic probes at `1.5e-4` and, only if appropriate, `6e-4`. Such probes do not silently replace the baseline; any replacement must be recorded as a new project decision.

### Schedule

The short failure-detection preflight uses constant `3e-4` so schedule effects do not obscure basic execution defects.

The longer qualification uses the trainer's token-count warmup/stable/cosine-decay schedule:

```text
schedule: WSD-shaped warmup / stable / cosine decay
warmup: max(16 optimizer updates, 5% of planned updates)
stable phase: remainder before final decay
cosine decay: final 20% of planned updates
minimum learning-rate ratio: 0.1
```

The scheduler is expressed and checkpointed in committed non-padding target tokens. Once the finite dataset token budget is selected, the launch preparation converts update counts and ratios into exact integer token horizons.

For a one-pass run of approximately 10M target tokens at 32,768 target tokens per update, the illustrative conversion is:

```text
planned updates: approximately 305
warmup: 16 updates = 524,288 target tokens
stable: approximately 228 updates = 7,471,104 target tokens
decay: approximately 61 updates = 1,998,848 target tokens
minimum LR: 3e-5
```

The exact final integer values must be computed from the verified manifest and frozen in the launch configuration. The example does not itself approve the 10M source-token envelope.

### Research anchors and interpretation

The baseline deliberately borrows stable, public structural choices rather than pretending frontier values transfer exactly:

- DeepSeek-V4, arXiv `2606.19348`: hybrid Muon + AdamW; AdamW betas `0.9/0.95`; weight decay `0.1`; Muon momentum `0.95`; Muon update RMS `0.18`; warmup, long stable phase, and cosine decay to one tenth of peak LR.
- Kimi K3, arXiv `2607.24653`: per-head Muon and a cosine schedule with linear warmup; its report does not disclose enough absolute values to copy a complete small-model recipe.
- 2026 norm-constrained optimizer work, including arXiv `2602.05813`, supports warmup followed by decay as a natural stability pattern for Muon-like optimizers.

The project does not adopt Kimi K3 per-head Muon in this qualification because that would change optimizer mechanics rather than merely select hyperparameters. It can be studied later as a separate optimizer experiment.

## Checkpoint, validation, and publication cadence

The provisionally approved cadence is:

```text
local joint checkpoint: every 25 successful optimizer updates
validation: every 50 successful optimizer updates
remote joint-checkpoint publication: every 50 successful optimizer updates
```

This cadence remains conditional on measured overhead. The intended aggregate recurring overhead budget is at most 5% of wall-clock training time. That number is provisional until timing is measured on the exact T4 path, then it is frozen in the qualification report.

Local checkpointing protects recovery and should remain at 25 updates unless it is itself unexpectedly expensive. Validation initially uses a small fixed slice. Remote publication should be asynchronous or deferred when the correctness contract permits; blocking upload time must be measured separately from background upload completion time.

If validation plus remote publication causes aggregate recurring overhead above the approved budget, those two operations may move to every 100 updates. Local recovery checkpoint cadence must not be widened automatically.

## Why thresholds are derived rather than guessed

There are three kinds of qualification threshold:

1. **Hard correctness gates** come from the project contract and allow no statistical interpretation.
2. **Hardware and operational gates** come from measurements on the exact T4, software stack, dataset geometry, and checkpoint path.
3. **Optimizer-stability gates** come from the observed distribution and repeatability of the standard baseline during controlled preflight runs.

A threshold must state:

- the metric;
- the measurement window;
- excluded warmup or startup samples;
- the statistic used, such as median, percentile, or median absolute deviation;
- warning and failure boundaries;
- the action taken on warning or failure;
- the exact run identity from which the threshold was derived.

Thresholds may not be reverse-engineered after seeing the final result. They are calculated from the preflight evidence, written into the qualification run summary, and frozen before the longer qualification segment starts.

## Instrumentation required before threshold calibration

The longer qualification may not begin until the trainer records at least:

```text
successful optimizer update number
committed target-token count
dataset block ID and manifest identity
training loss and smoothed loss
base LR and effective LR for each optimizer branch
GradScaler scale
whether an overflow occurred
whether an optimizer update was skipped
number of retry attempts
global pre-clip gradient norm
whether clipping occurred
per-optimizer-branch gradient norms
Muon aggregate and per-matrix update RMS
Muon update-to-weight ratio
AdamW update-to-weight ratio
CUDA allocated memory
CUDA reserved memory
step compute time
data-wait time
checkpoint save time and byte size
checkpoint load time
validation time and number of validation tokens
remote-publication blocking time and completion time
exact git commit, model config, optimizer recipe, routing identity,
dataset manifest/schema identity, and approved weight-file identity
```

Metrics must be written durably enough that an interrupted process does not erase the evidence needed to assess the preceding updates.

## Qualification stages

### Stage 0 — Offline and identity gates

Before using the T4:

- run the complete repository test suite on the exact launch commit;
- verify the finite qualification dataset with a full schema scan;
- verify `context_length=2048` and `sequences_per_block=16` in the manifest;
- verify the approved source revision, tokenizer identity, schema identity, and mixture-weight SHA-256;
- construct the hybrid optimizer and confirm every trainable parameter is routed exactly once;
- serialize and reload a synthetic checkpoint with model, optimizer, scheduler, scaler, RNG, counters, and dataset state.

Any failure blocks the T4 run.

### Stage 1 — Short standard-hyperparameter preflight

Run the standard baseline with constant `3e-4` for approximately 20 successful updates. This stage is designed to reveal immediate defects, not estimate model quality.

The first few updates are treated as startup observations. They remain logged, but throughput and steady-state distributions are calculated from the later portion of the stage.

Required observations include:

- finite forward loss;
- finite backward gradients;
- successful Muon and AdamW updates;
- stable or recoverable GradScaler behavior;
- clipping frequency and gradient distribution;
- update-RMS behavior around the configured Muon target;
- peak allocated and reserved memory;
- data-wait fraction;
- local checkpoint timing;
- one small validation timing measurement.

### Stage 2 — Uninterrupted baseline segment

Run at least 50 successful updates from a known initial checkpoint using the longer-run schedule or its precisely matched prefix. This segment establishes the reference trajectory and performance distributions.

Record medians, percentile ranges, and median absolute deviations after excluding the declared warmup/startup window.

### Stage 3 — Controlled interruption and local resume

Repeat from the same initial state and consume the same dataset blocks. Terminate the actual trainer process group at the planned checkpoint boundary, normally update 25, then resume from the local joint checkpoint.

Compare against the uninterrupted baseline:

- exact dataset cursor and next block;
- exact successful-update and committed-token counters;
- exact optimizer-group and parameter-routing identities;
- exact scheduler and scaler state;
- exact RNG state when supported by the deterministic path;
- first post-resume loss and subsequent trajectory;
- parameter, optimizer-state, and metric divergence.

The numerical tolerance is derived from an A/A repeatability control on the same T4. If two uninterrupted runs are bitwise identical, resume must also be bitwise identical. If the platform has a measured nondeterministic floor, the resume tolerance is frozen as a small multiple of that observed floor and may not be chosen after inspecting resume divergence.

### Stage 4 — Remote publication and empty-environment recovery

Publish a verified joint checkpoint, start from an environment without the prior local cache/checkpoint directories, restore the checkpoint and the required Drive cache window, and continue from the exact next block.

Remote success requires the same identity and trajectory checks as local resume, plus verified object IDs, sizes, and hashes. A remote object merely existing is not sufficient evidence.

### Stage 5 — Longer qualification segment

Only after Stages 0–4 pass, run the planned one-pass finite-dataset segment with the frozen hyperparameters, token horizons, cadence, and thresholds.

Run held-out validation and deterministic generation checks from trainer-produced checkpoints. The approximately-20M model is not expected to meet a substantive language-quality target; these checks prove functional model behavior and checkpoint usability.

## Hard correctness gates

The following are immediate failures:

- any non-finite loss accepted as a successful update;
- any non-finite gradient, optimizer update, model parameter, optimizer state, or scheduler state;
- exhausted overflow retries;
- any dataset block skipped, duplicated, consumed out of order, or acknowledged without a successful optimizer step;
- a checkpoint taken in the middle of an atomic prepared block;
- dataset manifest, schema, source revision, tokenizer, model, optimizer, routing, schedule, or weight-file identity mismatch;
- any trainable parameter omitted from optimizer routing or routed more than once;
- corrupt, partial, unverified, or configuration-incompatible checkpoint acceptance;
- resumed counters or dataset position that do not identify the exact next update;
- remote recovery that cannot verify every referenced immutable object;
- validation or generation producing non-finite model values;
- a process-level interruption test that terminates only a wrapper while the trainer continues.

Warnings cannot override a hard correctness failure.

## Provisional optimizer-stability gates

These gates are provisional until Stage 1 and Stage 2 generate the exact baseline distributions. Their final values are frozen before Stage 5.

### FP16 overflow and scaler behavior

Expected behavior:

- a small number of early scaler reductions may be recoverable during startup;
- the scaler must settle rather than decline continuously;
- successful updates must resume after any recoverable overflow.

Provisional warning:

- more than one overflow in a 25-update post-warmup window;
- any two consecutive skipped candidate updates;
- scaler scale falling across several checkpoints without recovery.

Provisional failure:

- retry budget exhausted for one prepared block;
- more than 1% skipped candidate updates after the declared warmup window;
- repeated consecutive overflows after warmup;
- persistent scaler collapse accompanied by non-finite gradients or no forward progress.

### Gradient clipping

Clipping is a safety mechanism, not automatically a defect. The relevant signal is persistent dependence on clipping.

Provisional warning:

- clipping on more than 20% of post-warmup successful updates in a rolling window;
- median pre-clip norm increasing materially across successive windows.

Provisional failure:

- clipping on more than 50% of post-warmup updates for a sustained window;
- gradient norms continue to grow while loss worsens;
- clipping is required on nearly every update to prevent non-finite execution.

The final boundaries are compared with the baseline distribution and may be tightened when the standard recipe is clearly stable.

### Muon and AdamW update behavior

The configured Muon target update RMS is `0.18`. Instrumentation must distinguish the configured normalization target from the actual effective parameter update after LR and weight decay.

Qualification examines:

- aggregate update RMS;
- per-matrix update RMS distribution;
- update-to-weight ratios by optimizer branch;
- persistent outliers by parameter role;
- whether resume restores the same optimizer-state-dependent update.

A single unusual matrix is investigated but is not automatically a run failure. Persistent branch-wide or role-wide deviation, non-finite update statistics, or a stepwise growth pattern correlated with loss instability is a failure.

Exact warning bands are derived from Stage 1 and Stage 2 because matrix shape and parameter role affect the observed distributions.

### Loss trajectory

The run does not need to achieve a predetermined quality score. It must show a finite, non-runaway optimization trajectory.

Required:

- every accepted update has finite loss;
- the smoothed post-warmup loss does not exhibit persistent explosive growth;
- the final smoothed window is not materially worse than the stable baseline without an explained schedule transition;
- validation loss is finite and checkpoints can reproduce it within the measured repeatability tolerance.

A provisional runaway detector should combine a relative increase and a robust baseline statistic, for example repeated losses above both a multiple of the stable median and a median-plus-MAD boundary. Its exact constants are calibrated before Stage 5 rather than hard-coded from intuition.

## Provisional hardware and operational gates

### Throughput

Define the steady-state throughput baseline as the median target tokens per second from the declared Stage-2 measurement window, excluding startup, validation, checkpoint, and data-wait intervals unless the metric explicitly measures end-to-end throughput.

Provisional failure threshold:

```text
steady-state compute throughput < 90% of the frozen baseline median
```

A drop is investigated before failure classification if caused by a known environment change such as power limits, thermal throttling, or another process.

### Data wait

```text
data-wait fraction target: < 5% of training wall time
```

The metric must separate waiting for the next prepared block from GPU compute and from checkpoint/validation time. A finite dataset completed before training should normally make this small. The later live producer/trainer concurrency test may use a separately measured threshold.

### Recurring operational overhead

```text
local checkpoint + validation + blocking publication overhead target: <= 5%
```

Background publication completion time is reported separately. The denominator and included operations must be stated in the report.

If the 25/50/50 cadence exceeds the frozen budget, validation and remote publication may widen to 100 updates. Local checkpoint cadence is changed only by an explicit decision.

### Memory margin

The run records both allocated and reserved CUDA memory. The provisional requirement is:

```text
at least 10% device-memory headroom,
preferably at least 1.5 GiB on the 16 GiB T4
```

Any out-of-memory event is a failure for the selected microbatch/profile. A lower but stable margin requires explicit review because validation, checkpoint staging, or later instrumentation may increase peak usage.

## Validation and generation gates

Validation uses a fixed, identified held-out slice. The report records its token count and block IDs. It must not overlap training data under the manifest split contract.

Validation qualification requires:

- finite loss;
- deterministic data selection;
- reproducible result from the same checkpoint within the measured numerical floor;
- no mutation of trainer counters, optimizer state, scaler state, or training data cursor;
- measured wall-clock cost.

Generation qualification uses a small fixed prompt set and deterministic greedy decoding. It requires:

- successful tokenization and semantic-vocabulary cropping;
- no non-finite logits;
- at least one generated semantic token when generation length permits;
- identical output from the same checkpoint and prompt under the deterministic path;
- successful generation after local and remote restoration.

Generated text quality is logged for inspection but is not a hard linguistic-quality gate at 20M parameters.

## Threshold calculation and freezing procedure

For each continuous metric:

1. Declare the startup/warmup exclusions before examining the final segment.
2. Record every sample, not only aggregates.
3. Calculate robust center and spread statistics, normally median, percentiles, and median absolute deviation.
4. Inspect time trends so a stable median cannot hide monotonic deterioration.
5. Compare uninterrupted A/A runs to establish the platform repeatability floor.
6. Define warning and failure boundaries from the measured distribution plus the hard operational needs.
7. Write the boundaries, formulas, measurement windows, and source run IDs into the qualification run summary.
8. Commit the resulting threshold table to `llm_docs/` before Stage 5.
9. Refuse threshold changes during Stage 5 unless the run is stopped and the change is recorded as a new experiment.

The baseline used to create thresholds must itself pass every hard correctness gate. A broken baseline cannot define acceptable behavior.

## Final qualification result

The approximately-20M T4 profile passes only when:

- every hard correctness gate passes;
- the standard hyperparameter baseline completes the required stages;
- final frozen optimizer-stability thresholds pass;
- final frozen hardware and overhead thresholds pass;
- local interruption/resume passes;
- remote publication and empty-environment recovery pass;
- validation and generation gates pass;
- the exact evidence and identities are preserved in a durable report;
- `project_status.md` is updated with the result and any deviations.

A warning may be accepted only with an explicit written rationale and without violating a hard gate. A failed gate cannot be waived by calling the approximately-20M model a smoke test.

## Decisions still open

This protocol does not silently decide:

- the finite qualification dataset source-token target, minimum, and maximum;
- whether the dataset is completed before training or later qualified in live producer/trainer overlap mode;
- shard size, prepared-block queue, and producer head-start settings;
- the exact fixed validation slice size;
- best-checkpoint metric;
- empty-environment cache prefetch window;
- whether one seed is sufficient after the first engineering pass;
- final empirically derived warning/failure numbers that require T4 measurements.

Those decisions must be added to project memory when made.