---
status: accepted
date: 2026-08-21
supersedes: 0099
---

# 0114 — Run the 100M/10B deep-decay continuation on one Modal H100

## Context and problem statement

ADR 0095 froze the scientific deep-decay trajectory. ADR 0099 retained that
schedule but moved execution to Kaggle two-T4 DDP and changed execution
microbatch four to two inside the same 64-sequence optimizer block. A verified
continuation checkpoint now exists in the shared Hugging Face namespace, and
the user has chosen the existing one-H100 Modal lane for the rest of the run.

The completed Modal 100M/2B trajectory already qualified microbatch 16 for the
same 100M model, FP16 precision, context 2,048, GDN-2 backend, and 64-sequence
block geometry. This makes 16 the concrete H100 execution slice; it is not a
new optimizer batch or scientific hyperparameter.

## Considered options

- Keep the active continuation on Kaggle two-T4 DDP.
- Return to a single Beam GPU.
- Resume the verified continuation on one exact Modal H100.

## Decision outcome

Chosen option: **resume on one exact Modal H100**, because the existing Modal
lane already qualified microbatch 16 for this model/block geometry and the user
explicitly authorized that provider. Supersede ADR 0099's Kaggle execution
choice while retaining the complete
ADR-0095 scientific schedule and checkpoint namespace.

The canonical command is:

```bash
modal run --detach modal/launch.py \
  --action deep-decay \
  --model 100M \
  --tokens 10B
```

The execution contract is:

- exactly one Modal H100 (`H100!`);
- one global 64-sequence optimizer block per update;
- execution microbatch 16, therefore four ordered accumulation slices;
- FP16, GDN-2, hybrid Muon+AdamW, model/optimizer/scaler/RNG state, data cursor,
  frozen validation prefix, and rolling 10B corpus order unchanged;
- the ADR-0095 cosine-settle, calibrated power-law, and terminal-linear
  schedule unchanged;
- W&B and Hugging Face run ID
  `100m-10b-deep-decay-from-step15500` unchanged;
- verified Hugging Face continuation state is preferred over the original
  source; the exact uncooled `100m-10b-data-001` step 15,500 is accepted only
  if the continuation namespace has no checkpoint;
- CPU restore, full checkpoint-state verification, dataset-window staging,
  and SHA verification must all complete before the H100 function is spawned;
- an existing continuation may be rewritten only in `microbatch_size` and the
  derived configuration hash. The exact source fallback may additionally
  receive the already-authorized ADR-0095 scheduler fields;
- remote checkpoints remain on a 250-successful-update cadence plus segment
  final, with no automatic H100 retry that could bypass CPU restaging.

The adapter must fail closed on a malformed or out-of-horizon pointer, wrong
data cursor, schedule or LR drift, model/optimizer/precision drift, missing
scaler or RNG state, non-authorized prior microbatch, stale staged window, or
non-finite update telemetry.

## Consequences

### Positive

- Kaggle two-T4 execution becomes historical provider evidence; its checkpoints
  remain exact continuation inputs after the execution-only `2 -> 16` rewrite.
- The existing Modal CPU-before-H100 staging and unified Hugging Face checkpoint
  transport can be reused without a second scientific trainer.

### Negative or limiting

- Floating-point reduction order changes again because one H100 uses four
  microbatch-16 slices instead of two ranks with sixteen microbatch-2 slices.
  The optimizer batch and ordered examples do not change.
- Modal's 24-hour function limit still makes explicit session slicing a normal
  recovery mechanism. Rerunning the same command restages from the newest
  durable checkpoint.

## Validation

Focused tests must reject schedule, LR, model, cursor, and unauthorized
microbatch drift and prove that an existing continuation rewrite changes only
the execution microbatch. The first live Modal segment must restore the newest
verified Hugging Face checkpoint, report its expected LR, complete finite H100
updates, and publish a later verified checkpoint under the unchanged namespace.

## Links

- [`0095-decay-1e-4-to-1e-5-then-5e-6.md`](0095-decay-1e-4-to-1e-5-then-5e-6.md)
- [`0099-run-deep-decay-100m-10b-on-kaggle-dual-t4.md`](0099-run-deep-decay-100m-10b-on-kaggle-dual-t4.md)
- [`../runbooks/100m_10b_deep_decay_modal.md`](../runbooks/100m_10b_deep_decay_modal.md)
- [`../../modal/deep_decay_10b_from_15500.py`](../../modal/deep_decay_10b_from_15500.py)
