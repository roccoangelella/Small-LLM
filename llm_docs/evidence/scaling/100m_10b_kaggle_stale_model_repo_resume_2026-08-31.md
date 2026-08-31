---
status: evidence
date: 2026-08-31
---

# 100M/10B Kaggle stale model-repository resume — 2026-08-31

## What was observed

A Kaggle two-T4 continuation showed `train/block_id=61571` even though the
preceding Modal segment had reached roughly step 70k. This was a real rewind,
not a display-only offset.

Authenticated W&B history for run
`100m-10b-deep-decay-from-step15500` contained:

```text
highest prior trainer/global_step:  70,291
matching train/block_id:            70,290
latest Kaggle trainer/global_step:  61,574
matching train/block_id:            61,573
```

`block_id` is zero-based and `trainer/global_step` is the count after the
successful update, so block 61,571 corresponds to global step 61,572.

The two Hugging Face latest pointers disagreed:

```text
checkpoint Storage Bucket latest:  step-00070250
legacy shared model-repo latest:    step-00061500
```

The private Bucket was
`roccoangelella/small-llm-100m-qualification-checkpoints`. Its authenticated
`latest.json` named
`run/100m-10b-deep-decay-from-step15500/checkpoints/step-00070250/last`.
The remote tree contained all five expected objects. The pointer's manifest
recorded the 913,885,860-byte `trainer_state.pkl` with SHA-256
`e53732922f4fcaae373c0b6bc581203b575d7aa634c6a7f113171b8bd503bd34`.
A fresh authenticated full download of that 913,885,860-byte object produced
the same SHA-256, not merely matching object metadata. The downloaded small
metadata also verified:

```text
checkpoint_id:                 step-00070250
optimizer_step_complete:       true
last_consumed_block_id:        70,249
validation loss:               2.830845956457779
validation target tokens:      2,097,152
source commit:                 61573f2248f397524c767c57beb57d9048bdd5ed
saved execution microbatch:    16
transport:                     modal-hf-bucket-checkpoint-v1
```

The highest W&B step 70,291 was not checkpoint-durable. Exact recovery must
therefore replay blocks 70,250 through 70,290 from Bucket step 70,250. Those 41
updates are the expected durability-window replay.

The dedicated best-model repository also existed and marker-verified
`step-00068250` at validation loss `2.824985434883274`. It is selection state,
not the rolling exact-resume source.

## Root cause

Commit `61573f2` had already moved Modal rolling `latest` writes from the shared
Git-backed model repository to the checkpoint Storage Bucket, but the committed
Kaggle deep-decay restore path still did this:

```text
runtime_base._hf_model_repo_store()
  -> run/100m-10b-deep-decay-from-step15500/latest.json
  -> step-00061500
```

It never compared the Bucket pointer. The model-repository pointer had remained
at step 61,500 after the earlier repository-quota incident, while Modal had
continued publishing to the Bucket through step 70,250. Kaggle therefore
restored valid but stale bytes and began duplicating old blocks. The observed
Kaggle updates after step 61,500 are not forward progress beyond the accepted
Bucket continuation.

## Repair

The Kaggle CPU gate now:

1. reads both checkpoint-Bucket and legacy model-repository pointers;
2. selects the highest step, preferring the Bucket on a tie;
3. downloads and manifest-verifies the selected checkpoint before GPU work;
4. applies only the authorized Modal/Beam-to-Kaggle execution rewrite
   (microbatch 16/4 to 2 and one-to-two CUDA RNG topology);
5. republishes and independently reads back the migrated bytes in the Bucket;
6. uses Bucket latest for subsequent rolling Kaggle publication and keeps the
   dedicated best-model repository separate.

A regression test freezes the observed `Bucket=step-00070250` versus
`legacy=step-00061500` case. The targeted Kaggle/transport test set passed.

## Operational consequence

The stale Kaggle process should be stopped before spending more T4 time on the
duplicate branch. After updating to a clean checkout containing the repair,
rerun the same canonical command. The expected exact restore is
`step-00070250`; after execution-topology migration, the first successful
Kaggle update should consume block 70,250 and report global step 70,251.
