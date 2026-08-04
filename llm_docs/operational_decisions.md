# Operational Decisions

_Last updated: 2026-08-03_

## 2026-08-02 — Launch the authenticated 10M-token dataset pilot

The user authorized running the bounded 10M-token dataset acceptance pilot on the VPS through PiLink.

Execution contract:

- the VPS checkout must be clean and exactly match the repository `main` branch before launch;
- use the approved exact ClimbMix weight artifact with SHA-256 `76e82e22760adcac59c7294fe9bac11358f5a8b7a26035aae64c3f2e6fa1acb7`;
- use the committed authenticated pilot bounds: 10,000,000 target accepted source tokens, 9,000,000 minimum, 11,000,000 hard maximum, and 2,000,000-token durable checkpoint cadence;
- require the real personal-Google-Drive mirror and authorized-user OAuth credentials;
- do not substitute `--allow-local-only` or a synthetic smoke run for the authenticated acceptance pilot;
- retain durable logs, exit codes, interruption snapshot, completed-resume baseline, manifests, hashes, and acceptance reports under the documented `/data/climbmix-*` paths;
- complete the intentional interruption/resume, schema-v2 full verification, completed-resume idempotence, and fail-closed acceptance verification before declaring the pilot passed.

Launch is permitted only after the current checkout's operational prerequisites are present and validated, including the approved calibration artifact, sufficient disk, remote dependencies, passing offline evidence, OAuth token, and Drive folder configuration.

## 2026-08-02 — Authenticated 10M-token dataset pilot passed

The authorized pilot completed successfully at repository commit `e4776501d68e39746f8a75dcbb9c49515f215abd`. The accepted run used the approved exact mixture file, real personal-Google-Drive mirroring, and the committed 10M/9M/11M bounds with a 2M-token checkpoint cadence.

The accepted evidence records:

- a durable incomplete snapshot at 2,000,112 accepted source tokens and 2,814 documents;
- termination of the actual producer process group with exit status 143;
- successful continuation with the identical production command plus `--resume`;
- final completion at 10,000,662 accepted source tokens and 14,136 documents;
- seven local immutable shards and seven matching Drive entries;
- successful full schema-v2 verification;
- successful completed-resume idempotence;
- fail-closed acceptance status `PASS` for environment, calibration, offline tests, calibration run, Drive smoke, pilot, interruption/resume, and completed-resume idempotence.

The canonical acceptance report is `/data/climbmix-ops/dataset_acceptance_report.json`, SHA-256 `b18decde4aa0e6e7376c3fecd3dda4406dee983f11224537cf73dd22a66bc00b`.

A prior attempt that did not terminate the actual producer was explicitly rejected, archived, and excluded from the accepted evidence. This prevents a wrapper-level signal from being misrepresented as an interruption/resume qualification.

## 2026-08-03 — Freeze accepted interruption evidence and pilot interpretation

The 10M pilot exposed a distinction between terminating a wrapper and terminating the dataset producer. The first orchestration attempt signalled only its wrapper shell; the child producer continued, retained the production lock, and completed. That attempt is invalid as interruption evidence, was archived for forensics, and is excluded from all accepted reports.

The operational contract is now:

- run the producer in a dedicated process group or equivalent supervised unit;
- capture an incomplete snapshot only after every referenced shard is remotely durable;
- terminate the complete producer group, wait for exit, and confirm no descendant or lock holder remains;
- launch `--resume` only after those checks;
- reject wrapper exit codes, snapshots, or lock conflicts as proof by themselves;
- use `uv run python` or the project interpreter rather than assuming a system `python` executable.

The pilot also freezes the following interpretation boundaries:

- its 10,000,662 accepted tokens, seven local/Drive shards, and 119-second end-to-end acceptance sequence qualify operational correctness, not 90B throughput;
- only seven of nineteen accepted clusters appeared at 10M tokens, with normalized mixture error `0.08533077992520376` while the accepted command used the permissive production default `maximum_rolling_mixture_error=1.0`, so the pilot is not a representative all-cluster training sample;
- source-reader resume currently replays prior documents to verify the durable cursor, creating restart work that grows with cursor depth and must be qualified before full production;
- current production disk preflight requires about 222.3 GiB for 90B and 247.0 GiB for the 100B hard maximum, while the pilot VPS had about 95 GiB free; full production therefore requires more capacity or a proven bounded-cache eviction lifecycle.

These findings do not revoke the pilot pass. They define new preconditions for the later 90B launch and prevent bounded operational evidence from being overgeneralized.

## 2026-08-03 — Freeze the initial T4 training block at 16 sequences

The user selected `sequences_per_block=16` for the first approximately-20M training-qualification cache on the NVIDIA T4.

Previous state:

- the authenticated dataset pilot inherited the production-cache default of 512 sequences per block;
- the trainer later defined one durable prepared block as one atomic optimizer update;
- those independent choices implied approximately 1,048,576 target tokens per update at context 2,048, but no user decision had approved that as training geometry.

New decision:

```text
context_length: 2,048
sequences_per_block: 16
microbatch_size: 1
effective target tokens per update: approximately 32,768
```

Reasoning and evidence:

- the corrected T4 benchmark measured approximately 1,291 target tokens/s for the selected FP16 GDN-2 chunk-32 path;
- a 16-sequence block therefore gives a roughly 25-second update before data/checkpoint overhead;
- a 10M-train-token cache would provide approximately 305 updates instead of only 9–10;
- this gives enough cadence to observe loss, scaler behavior, clipping, Muon statistics, checkpointing, interruption, and resume;
- microbatch size remains 1, so this decision does not increase per-forward activation memory.

Implementation boundary:

- do not modify the global dataset-production default of 512 sequences per block;
- build a separate finite training-qualification dataset with explicit `--sequences-per-block 16`;
- launch the trainer with explicit `--sequences-per-block 16`, so the manifest identity check rejects a wrong dataset;
- preserve the accepted 512-block pilot unchanged as operational dataset evidence;
- consider 32 sequences per block only as a later measured batch-growth comparison after the 16-sequence profile passes.

The source-token target, shard size, queue/head-start settings, learning rates, schedule, checkpoint/evaluation cadence, acceptance thresholds, and number of seeds remained open at this point.

## 2026-08-03 — Provisionally approve the first T4 checkpoint and evaluation cadence

The user approved the proposed qualification cadence provided that it does not materially slow training:

```text
local joint checkpoint: every 25 successful optimizer updates
validation: every 50 successful optimizer updates
remote joint-checkpoint publication: every 50 successful optimizer updates
```

This is a conditional operational decision, not yet an unconditional launch constant. Before the longer qualification segment, the preflight must measure checkpoint save time, validation time, remote-publication time, and their aggregate wall-clock fraction.

Implementation and interpretation boundaries:

- local checkpointing remains at 25 updates unless it is itself unexpectedly expensive;
- validation should initially use a small fixed validation slice so its cost is measurable and comparable;
- remote publication should not block the training process longer than the accepted overhead budget; asynchronous or deferred publication is preferred when correctness permits it;
- the intended aggregate recurring overhead budget is 5% of wall-clock training time, subject to confirmation on the exact T4 path;
- if the measured cadence exceeds the frozen overhead budget, validation and remote publication may be moved to a wider interval without changing the optimizer, update batch, or local recovery cadence.

The number of validation blocks, best-checkpoint metric, and remote prefetch window remain open.

## 2026-08-03 — Replace unclear “bounded cache” terminology

The user reported that the phrase “bounded cache” was unclear. Project documentation should now prefer **finite qualification dataset** and explain the mechanism whenever the old term appears in historical context.

A finite qualification dataset means:

- a dataset build with an explicit accepted-source-token target, minimum, and hard maximum;
- immutable tokenized schema-v2 shards consumed by the trainer;
- a defined completion point far below the full 90B production target;
- no implicit relationship to context length, optimizer batch size, or epoch count.

The source-token envelope controls how much distinct source material is prepared. The trainer's number of passes controls how many times that prepared material is consumed. They must be configured separately, and the first qualification should not silently repeat the dataset.

This terminology decision does not approve a source-token envelope. The previously proposed 10M/9M/11M training envelope remains open until explicitly selected.

## 2026-08-03 — Use a conservative standard hyperparameter baseline

The user decided not to conduct a broad hyperparameter search for the approximately-20M engineering qualification. The run will use a conservative, reproducible baseline drawn from conventional decoder pretraining and the public structure of post-2025 Muon recipes.

Frozen baseline:

```text
base AdamW learning rate: 3e-4
AdamW beta1 / beta2: 0.9 / 0.95
AdamW epsilon: 1e-8
AdamW weight decay: 0.1
Muon momentum: 0.95
Muon LR multiplier: 1.0
Muon target update RMS: 0.18
Muon weight decay: 0.1
global gradient clipping norm: 1.0
```

Schedule policy:

- short failure-detection preflight: constant `3e-4`;
- longer qualification: token-count warmup/stable/cosine decay;
- warmup: the larger of 16 optimizer updates or 5% of planned updates;
- cosine decay: final 20% of planned updates;
- stable phase: all remaining updates between warmup and decay;
- minimum LR ratio: `0.1`.

The exact token horizons are derived from the verified finite-dataset manifest after the source-token envelope is selected. They are then checkpointed and frozen before the longer run.

There is no planned broad LR sweep. If the baseline fails a hard stability gate, narrow diagnostic probes at half and, only if justified, twice the baseline LR may be used. Any replacement is a new decision and cannot be inferred from a diagnostic result.

Research interpretation:

- DeepSeek-V4 (`2606.19348`) publicly supports the selected optimizer split, `0.9/0.95` AdamW betas, `0.1` weight decay, `0.95` Muon momentum, `0.18` update RMS, and a warmup/stable/cosine structure.
- Kimi K3 (`2607.24653`) supports cosine scheduling and Muon-family optimization but does not disclose a complete set of absolute values suitable for copying.
- Kimi K3 per-head Muon is not adopted here because it is an optimizer-mechanics experiment, not a standard hyperparameter choice.

The exact rationale and launch conversion rules are in `20m_qualification_protocol.md`.

## 2026-08-03 — Approve empirical threshold derivation and fail-closed qualification gates

The user approved deriving optimizer, hardware, overhead, and resume thresholds from controlled T4 evidence rather than inventing arbitrary final numbers or copying frontier-scale limits.

The approved method is:

1. install the required instrumentation;
2. run the standard short preflight;
3. run an uninterrupted reference segment;
4. run an A/A repeatability control on the same T4;
5. run a controlled local interruption/resume comparison;
6. run verified remote publication and empty-environment recovery;
7. calculate robust baseline statistics and the platform numerical-repeatability floor;
8. write warning/failure formulas and windows into project memory before the longer segment;
9. freeze those thresholds for the qualification run.

Hard correctness gates are not empirical and allow no waiver. These include finite arithmetic, complete/exclusive optimizer routing, exact identity matching, no skipped or duplicated data blocks, checkpoint integrity, exact next-block/counter restoration, and verified remote objects.

Provisional measurement targets include:

```text
steady-state throughput: at least 90% of frozen T4 baseline median
data-wait fraction: below 5%
recurring checkpoint/validation/blocking-publication overhead: at most 5%
T4 memory headroom: at least 10%, preferably at least 1.5 GiB
post-warmup skipped updates: no more than 1%
no exhausted overflow retry budget
clipping above 20% of post-warmup updates: warning
sustained clipping above 50%: failure
```

These continuous-metric numbers remain provisional until the preflight establishes valid measurement windows and distributions. They are not allowed to weaken any hard correctness gate.

The comprehensive metric definitions, stage structure, threshold lifecycle, resume-tolerance method, validation/generation gates, and pass criteria are recorded in `20m_qualification_protocol.md`.