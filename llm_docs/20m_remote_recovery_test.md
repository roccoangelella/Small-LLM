# Approximately-20M Remote Empty-Environment Recovery Test

_Last updated: 2026-08-04_

## Decision

The local interruption/resume gate passed exactly. The final pre-training qualification gate will therefore isolate private remote publication and empty-environment recovery rather than repeat another complete 50-update reference/A-A test.

The authoritative Kaggle entrypoint is:

```text
kaggle/run_20m_remote_recovery_from_clone.py
```

## Why this test is shorter

A further complete 50-update or 50+50 run is not required. The existing evidence already establishes:

- stable execution through 50 updates;
- exact same-T4 A/A repeatability;
- exact update-25 local interruption and resume;
- exact semantic checkpoint restoration through update 50;
- bounded accepted gradient clipping and zero FP16 overflows.

The remaining uncertainty is specifically remote durability and restoration. The new test therefore uses:

```text
publisher segment: updates 1-25
local reference continuation: updates 26-30
remote-restored continuation: updates 26-30
total executed training updates: 35
```

Five post-restore updates are sufficient to prove that model, optimizer, scheduler, scaler, RNG, counters, dataset cursor, and shard data all restore coherently. The test compares every recorded non-runtime numerical value and the complete semantic step-30 checkpoint state. It is not intended to re-estimate longer-run stability.

## Scope

One invocation performs this fail-closed sequence:

1. Keep the controlling repository on `main` and create a clean detached worktree at launch commit `45d1da4a1ac3f18cf6ce02b8439672f10e2c8b4c`.
2. Select the accepted mounted qualification dataset by both frozen manifest hashes.
3. Repeat the literal full dataset scan and exact 306-update plan reproduction.
4. Train from initialization through update 25 and create `step-00000025`.
5. Publish the complete verified checkpoint tree and embedded Drive manifest to a private Hugging Face checkpoint repository through the two-phase publisher.
6. Require a valid remote `latest.json` pointer to the verified update-25 snapshot.
7. Resume the local update-25 checkpoint for updates 26-30 to establish the reference continuation.
8. Delete the Hugging Face client cache and create a fresh destination containing no checkpoint or data cache.
9. Download the checkpoint from the private remote pointer and verify all checkpoint manifests and hashes.
10. Download exactly two train shards from Google Drive using the immutable file IDs, byte sizes, and SHA-256 values in the restored Drive manifest.
11. Require the restored cursor to identify block 25 as the next unconsumed block.
12. Compare the source and remotely restored update-25 semantic checkpoint state exactly.
13. Resume only from the restored checkpoint and restored two-shard cache for updates 26-30.
14. Compare the local and remote-restored trajectories exactly.
15. Compare local and remote-restored update-30 semantic checkpoint state exactly.
16. Preserve logs, exit codes, W&B run identities, remote pointer evidence, downloaded-shard identities, checkpoint comparisons, and a final summary JSON under `/kaggle/working`.

The mounted Kaggle dataset is used only to produce the source checkpoint and local reference. The remote-restored continuation points to the fresh restored cache and the restored checkpoint's `drive_manifest.json`, not the mounted dataset.

## Required Kaggle secrets

```text
WANDB_API_KEY
HF_TOKEN
SMALL_LLM_HF_REPO_ID
GOOGLE_DRIVE_OAUTH_TOKEN_JSON
```

Optional:

```text
WANDB_ENTITY
```

`SMALL_LLM_HF_REPO_ID` must be a private Hugging Face model repository ID such as `owner/small-llm-checkpoints`. The launcher may create it when the token has permission.

`GOOGLE_DRIVE_OAUTH_TOKEN_JSON` must contain the full authorized-user OAuth token JSON previously produced by the project's Drive authorization flow, including its refresh token. It is written to a mode-0600 temporary file inside the evidence directory and is never included in the final summary.

## Pass requirements

```text
published checkpoint: step-00000025
remote latest pointer: verified
empty restore destination: confirmed
prefetched Drive shards: exactly 2
prefetched shard hashes and sizes: exact
restored next block: 25
source vs remote step-25 semantic state: exact
local continuation steps: 26-30 exactly
remote continuation steps: 26-30 exactly
local vs remote non-runtime numerical differences: 0
local vs remote discrete trajectory: exact
local vs remote step-30 semantic state: exact
non-finite values: none
```

Raw checkpoint-tree byte identity is recorded but is not required when decoded semantic state is exact, consistent with the accepted serialization-byte finding from the A/A and local-resume tests.

## Successful result

```text
status: passed_remote_empty_environment_recovery
authorization: full_306_run_ready_for_explicit_launch
```

A pass completes the pre-training engineering qualification ladder and permits an explicit decision to launch the frozen 306-update one-pass run. It does not launch that run automatically.

## Kaggle invocation

```python
%cd /kaggle/working/Small-LLM
!git pull --ff-only
!python kaggle/run_20m_remote_recovery_from_clone.py
```

The authoritative final summary is:

```text
/kaggle/working/small_llm_remote_recovery_summary.json
```