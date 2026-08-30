---
status: evidence
observed_at: 2026-08-30
run_id: 100m-10b-deep-decay-from-step15500
---

# 100M/10B deep-decay Modal resume from step 59,750

## Result

The existing deep-decay trajectory resumed in the new Modal workspace from the
newest Hugging Face continuation, `step-00059750`. The detached live app is:

```text
profile: generico-podcast1
environment: main
app: ap-86sjxvNMobYbQ9pTYhfsZw
controller source: 115769ada2324025a190653a486d47b8b19ea9ee
```

The controller checkout was clean. The Modal deep-decay implementation in that
checkout is unchanged since the previously qualified `c57a9b3` adapter commit.
The first dry run stopped before allocation because the new workspace did not
yet contain `small-llm-training`. The secret was created with exactly
`HF_TOKEN`, `WANDB_API_KEY`, and `SMALL_LLM_HF_REPO_ID`; no credential value was
written to the repository. The repeated dry run then resolved the frozen
`modal_single_h100_block64` contract successfully.

## CPU restore and cross-provider migration

Before H100 dispatch, the CPU gate reported:

```text
checkpoint:          step-00059750
completed steps:     59,750
committed targets:   7,831,552,000
required data block: 59,750
expected LR:         1.3927322119926431e-05
remaining updates:   16,544
final step:          76,294
checkpoint source:   Hugging Face continuation namespace
```

The gate verified the remote pointer, manifests, checkpoint payload, frozen
schedule and expected LR, model/optimizer/scaler/RNG/cursor state, and staged
the checkpoint-aligned dataset window. It then performed only the authorized
execution migration:

```text
microbatch:       2 -> 16
global block:     64 sequences, unchanged
CUDA RNG states:  2 -> 1, byte-identical rank-zero state selected
hidden backup:    .step-00059750.pre-modal-h100
```

This directly exercises the Kaggle/Beam/Hugging-Face-to-Modal continuation
path. The original two-rank tree remains in the hidden provider-migration
backup. The CPU gate returned `ready` before the exact `H100!` function was
spawned.

## First live updates

The H100 worker restored W&B with `resume=must` under the unchanged run ID and
started with FP16, GDN-2 chunk 32, hybrid Muon+AdamW, microbatch 16, and four
ordered slices per 64-sequence update. The first resumed update was finite:

| step | block | consumed targets | loss | grad norm | LR | overflow retries |
|---:|---:|---:|---:|---:|---:|---:|
| 59,751 | 59,750 | 7,831,683,072 | 2.866167 | 0.621489 | 1.3926942873507823e-05 | 0 |

After first-step compilation, updates reached approximately 65k target
tokens/s. Finite telemetry was observed through at least step 59,770; the
step-59,770 row reported loss `2.936666`, gradient norm `0.689933`, LR
`1.392353040594126e-05`, throughput `65,023` target tokens/s, and zero overflow
retries. The client then disconnected with Modal's detached-app confirmation;
an authenticated app listing showed one live detached task.

`step-00059750` remains the newest observed durable HF pointer in this record.
The live worker's next normal validation and remote publication boundary is
step 60,000.

## Verification scope

No bounded H100 test segment and no broad unit-test suite were run. The actual
production CPU gate supplied the checkpoint/state/data verification, and the
first finite production updates supplied the hardware integration check. Four
focused memory-layout/link checks passed. The aggregate project-memory module
remains red on its existing ADR-shape and returned-legacy-path baseline; none of
those failures concerns this launch or the new evidence link.
