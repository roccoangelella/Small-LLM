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
- build a separate bounded training-qualification cache with explicit `--sequences-per-block 16`;
- launch the trainer with explicit `--sequences-per-block 16`, so the manifest identity check rejects a wrong cache;
- preserve the accepted 512-block pilot unchanged as operational dataset evidence;
- consider 32 sequences per block only as a later measured batch-growth comparison after the 16-sequence profile passes.

The bounded source-token target, shard size, queue/head-start settings, learning rates, schedule, checkpoint/evaluation cadence, acceptance thresholds, and number of seeds remain open.

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
- the numeric definition of “materially slow” remains to be approved after the first timing measurement;
- if the measured cadence exceeds the approved overhead budget, validation and remote publication may be moved to a wider interval without changing the optimizer, update batch, or local recovery cadence.

The number of validation blocks, exact overhead budget, best-checkpoint metric, and remote prefetch window remain open.