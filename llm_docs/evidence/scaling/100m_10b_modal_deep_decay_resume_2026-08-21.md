# 100M / 10B deep-decay Modal resume — 2026-08-21

## Result

The authorized `100m-10b-deep-decay-from-step15500` trajectory is running on
one exact Modal H100 from the newest verified continuation checkpoint available
at launch, `step-00027750`. The provider adapter changed only execution slicing and CUDA
RNG topology cardinality: the global optimizer block remains 64 sequences,
microbatch 16 gives four ordered slices, and the byte-identical rank-zero CUDA
RNG state is the live one-device state. The original two-rank checkpoint tree
is retained in the hidden provider-migration backup.

The live source commit is `c57a9b3ce3aedcdefd445cb6f664caf00ececfa0`. The
Modal app is `ap-jcW589PrB43z2tOdfeLOpS`:

```text
https://modal.com/apps/roccoangelella-59690/main/ap-jcW589PrB43z2tOdfeLOpS
```

The app remains intentionally uncapped. Rerunning the same command after a
normal Modal session boundary resolves the newest verified continuation again.

## Selected checkpoint and CPU gate

The authenticated Hugging Face pointer in
`roccoangelella/small-llm-100m-qualification` resolved to:

```text
checkpoint:       step-00027750
prefix:           run/100m-10b-deep-decay-from-step15500/checkpoints/step-00027750/last
completed steps:  27,750
committed targets: 3,637,248,000
last data block:  27,749
expected LR:      4.850625030893043e-05
validation loss: 2.9145127
trainer_state:    913,855,139 bytes
trainer_state SHA-256:
  42be81d87ac10dc0a2b81c8feeeaa291f670706824328b7d6affcab8d317981f
```

The pointer, two-phase checkpoint manifests, file sizes and hashes, run ID,
rolling-dataset identity, full trainer configuration, scheduler committed
targets, optimizer/model/scaler state, RNG state, and cursor were checked on a
CPU function before H100 dispatch. The gate reported the exact expected LR
above and selected the continuation rather than the original step-15,500
fallback. The staged dataset began at block 27,750.

The downloaded checkpoint already recorded Modal microbatch 16 on the final
launch because the preceding recovery attempt had installed the verified
provider-migrated copy. Its hidden backup preserves the original Kaggle
microbatch-2 configuration and two CUDA generator states.

## Live restore and update evidence

The H100 worker invoked the trainer with:

```text
resume checkpoint:     step-00027750
device:                NVIDIA H100 80GB HBM3
world size:            1
global block:          64 sequences / 131,072 targets
execution microbatch:  16 (four ordered slices)
precision/backend:     FP16 / GDN-2
optimizer:             hybrid Muon+AdamW
W&B/HF run ID:         100m-10b-deep-decay-from-step15500
remaining plan:        48,544 updates, through step 76,294
```

Checkpoint loading and W&B `resume=must` completed before the first update.
The first update included CUDA compilation; subsequent updates settled near
66k target tokens/s. Representative raw trainer telemetry:

| step | block | committed targets | loss | grad norm | LR | target tok/s | scaler | overflow retries |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 27,751 | 27,750 | 3,637,379,072 | 2.932615 | 0.634032 | 4.8503406401314246e-05 | 687 | 131,072 | 0 |
| 27,752 | 27,751 | 3,637,510,144 | 2.963849 | 0.561065 | 4.850056276290415e-05 | 41,395 | 131,072 | 0 |
| 27,755 | 27,754 | 3,637,903,360 | 3.035658 | 0.592205 | 4.849203346255866e-05 | 66,074 | 131,072 | 0 |
| 27,790 | 27,789 | 3,642,490,880 | 3.000492 | 0.538551 | 4.8392703637973666e-05 | 66,067 | 131,072 | 0 |
| 28,000 | 27,999 | 3,670,016,000 | 2.869480 | 0.593944 | 4.78035628434232e-05 | 66,324 | 131,072 | 0 |

Each recorded loss, gradient norm, LR, throughput, scaler value, and optimizer
update statistic was finite. The reported LR at every row exactly matched the
frozen ADR-0095 formula evaluated at that row's committed-target count. There
were no overflow retries and the scaler remained continuous at 131,072.

## First post-migration checkpoint

At step 28,000 the worker completed the frozen 16-block validation prefix at
loss `2.91551074560266` / perplexity `18.45823745925412`, then wrote a
913,886,745-byte local checkpoint in 6.43 seconds. The two-phase Hugging Face
publication completed in 29.35 seconds and only then pruned the rolling
step-27,750 tree. Training immediately continued with finite updates.

An independent authenticated read of the remote `latest.json` resolved to:

```text
checkpoint: step-00028000
prefix:     run/100m-10b-deep-decay-from-step15500/checkpoints/step-00028000/last
trainer_state.pkl: 913,885,544 bytes
trainer_state SHA-256:
  117cd17b551e97417db28bce6ad74382a842a8b6e12afc0f7e8594af6c2e1ba4
```

The remote pointer carries all four expected file identities and hashes. This
is the first durable checkpoint produced after the Kaggle-to-Modal execution
migration and is now the newest fail-closed resume point.

## Fail-closed incidents during migration

Two pre-update failures were useful migration evidence and did not advance the
data cursor:

1. App `ap-zXNwYmCGXWB5Ts4SlNH2IF` restored the two-T4 checkpoint but the
   single-GPU loader rejected its second CUDA RNG device entry. The adapter was
   changed to preserve the original tree and project only the byte-identical
   rank-zero CUDA RNG state onto the one H100. Tests reject every other
   topology.
2. App `ap-qgcGcmhk1hYVC4HdxksyJm` then stopped in the CPU gate because a
   descriptive execution-contract string differed from the already-persisted
   migrated contract. The string was restored rather than weakening the
   immutable comparison.

Both stops happened before an optimizer update. The successful app uses the
committed fixes in `3b18eb3` and `c57a9b3`.

## Implementation and verification

The implementation landed in commits `749d315`, `3b18eb3`, and `c57a9b3` and
was pushed to `origin/main`. Focused verification passed 24 tests covering the
Modal deep-decay adapter, Modal packaging/dispatch, rolling data, trainer
configuration, and exact resume. Python compilation and `git diff --check`
also passed.

The prescribed repository-wide unittest discovery ran 506 tests and remains
red for the already-documented unrelated baseline: 69 failures, 9 errors, and
1 skip. The failures include unavailable `pytest` imports, stale eval/ADR
expectations, a legacy dataset path, and an older remote-state equality test.
No focused Modal deep-decay test failed.
