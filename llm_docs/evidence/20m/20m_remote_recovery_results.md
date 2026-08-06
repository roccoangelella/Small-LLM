# Approximately-20M Remote Empty-Environment Recovery Results

_Last updated: 2026-08-05_

## Verdict

The final remote recovery qualification gate passed.

```text
status: passed_remote_empty_environment_recovery
authorization: full_306_run_ready_for_explicit_launch
resume class: exact_remote_empty_environment_recovery
```

This result completes the pre-training engineering qualification ladder for the frozen approximately-20M run. It permits an explicit launch decision for the complete 306-update one-pass segment; it does not launch that segment automatically.

## Frozen execution identity

```text
launch commit: 45d1da4a1ac3f18cf6ce02b8439672f10e2c8b4c
controller commit: cadea536b54a66e2f623c70f5ac41b077996d51d
worktree: clean and detached
GPU: Tesla T4, 15,360 MiB
started UTC: 2026-08-05T07:30:59.631323+00:00
finished UTC: 2026-08-05T07:50:51.749499+00:00
evidence directory: /kaggle/working/small-llm-remote-recovery-controller/small-llm-remote-recovery-20260805T073059Z
summary: /kaggle/working/small_llm_remote_recovery_summary.json
```

The mounted qualification dataset again passed the literal full scan and exact one-pass-plan reproduction.

```text
manifest SHA-256: 1e5ee8f372b77b6728288610dbe7cce74d833be21e53d1538bc5a890229b18bb
Drive manifest SHA-256: fbb29ee0d0102658e1274e39d6647cf56a6dcb685e0f566b1736847dcc4fbe84
Drive run ID: 20m-qualification-dataset-001
```

## Bounded test budget

The test isolated remote durability and restoration rather than repeating the already-passed 50-step stability measurements.

```text
publisher segment: updates 1-25
local reference continuation: updates 26-30
remote-restored continuation: updates 26-30
total executed training updates: 35
```

## Remote checkpoint publication

The publisher produced and uploaded `step-00000025` to the private Hugging Face checkpoint repository.

```text
checkpoint ID: step-00000025
remote pointer: run/20m-qualification-dataset-001/latest.json
remote last prefix: run/20m-qualification-dataset-001/checkpoints/step-00000025/last
publication event elapsed time: 14.691478 seconds
```

The published checkpoint manifest contained:

```text
checkpoint.json: 798 bytes
Drive manifest: 10,979 bytes
local manifest: 280 bytes
trainer state: 216,852,174 bytes
```

The source and remotely downloaded step-25 checkpoint trees were byte-identical:

```text
source tree SHA-256: 11ae85b4f31f221c821d87cd19aef1335e8468c5022fd633e21a012a23e15a59
remote tree SHA-256: 11ae85b4f31f221c821d87cd19aef1335e8468c5022fd633e21a012a23e15a59
```

## Empty-environment restoration

The restore destination was created empty. The remote checkpoint and exactly two train shards were downloaded and verified.

```text
destination was empty: true
last consumed block: 24
next block: 25
prefetched shard count: 2
```

Restored shards:

```text
train/train-000000.bin
size: 3,081,696 bytes
SHA-256: 19239d66dee6feae1c40306ffca38d1c7f71095b1cc834303a09f9cd27919ed6

train/train-000001.bin
size: 3,868,512 bytes
SHA-256: 63df284e162eb9865aad6bce4285166b2d2060683f871b176f4a254d564d0ecf
```

## Exact checkpoint semantics

The source and remote step-25 checkpoints matched exactly after decoding all state:

```text
tensors compared: 383
tensor elements: 54,184,616
containers: 189
scalars: 1,112
differences: 0
semantic exact: true
```

After both continuations reached update 30, the local and remotely restored checkpoints also matched exactly under the same semantic comparison:

```text
tensors compared: 383
tensor elements: 54,184,616
containers: 189
scalars: 1,112
differences: 0
semantic exact: true
```

The raw step-30 tree hashes differed while semantic state was exact:

```text
local step-30 tree SHA-256: e255e3e006e7f681cf287ab13bebbcf8176ba7205643fa068c251518a390886c
remote step-30 tree SHA-256: dc769476c9333ae1e61fe273744111b238db4cc8a131082c42108e6aaf9407b1
```

This is consistent with the previously accepted serialization-byte variability and is not state divergence.

## Exact resumed trajectory

The local and remote-restored continuations covered the same updates 26-30 and matched exactly.

```text
compared numerical values: 1,065
differing numerical values: 0
maximum absolute difference: 0.0
maximum relative difference: 0.0
numeric trajectory exact: true
discrete trajectory exact: true
```

The result proves coherent restoration of model, optimizer, scheduler, GradScaler, RNG, counters, dataset cursor, checkpoint manifests, and required dataset shard bytes.

## W&B run identities

```text
publisher: 20m-t4-remote-20260805-073105-publisher
local reference: 20m-t4-remote-20260805-073105-local-reference
remote restored: 20m-t4-remote-20260805-073105-remote-restored
project: Small-LLM
entity observed in prior Kaggle logs: rocchissimo936-none
```

These run identities should be preserved with the Kaggle evidence. The authoritative pass/fail verdict is the fail-closed summary and locally hashed evidence, not visual inspection alone.

## Qualification conclusion

Passed qualification ladder:

1. offline suite and corrected T4 kernel harness;
2. exact dataset identity, full scan, and plan reproduction;
3. 20-update integrated trainer preflight;
4. exact 50-update same-T4 A/A repeatability;
5. actual-process SIGTERM and exact local resume;
6. verified private checkpoint publication;
7. empty-environment checkpoint and data restoration;
8. exact remote-restored trajectory and semantic checkpoint equality.

The frozen complete one-pass run is now ready for an explicit launch decision:

```text
updates: 306
train target tokens: 10,006,528
schedule: WSD
warmup: 16 updates
stable: 228 updates
decay: 62 updates
minimum LR ratio: 0.1
```

No architecture, optimizer, LR, clipping, schedule, seed, dataset order, or initialization change is implied by this authorization.
