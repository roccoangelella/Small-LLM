---
date: 2026-08-18
status: observed-and-fixed
run_id: 100m-10b-deep-decay-from-step15500
---

# 100M/10B Kaggle deep-decay provider microbatch migration

## Observed first Kaggle launch

The first Kaggle dual-T4 launch restored the existing Hugging Face deep-decay checkpoint successfully (the large `trainer_state.pkl` download completed), then failed before training in `kaggle/deep_decay_10b_from_15500.py` with:

```text
RuntimeError: deep-decay checkpoint scientific/execution config drifted: {"microbatch_size": [4, 2]}
```

This proves the failure was not checkpoint download corruption or dataset staging. The restored deep-decay checkpoint still recorded the single-GPU execution microbatch (`4`), while the new Kaggle lane requires the already-authorized T4-safe execution microbatch (`2`).

## Root cause

ADR 0099 classifies `4 -> 2` as an execution-slicing migration while keeping the frozen 64-sequence global optimizer block unchanged. The first Kaggle verifier nevertheless required an already-published deep-decay checkpoint to have `microbatch_size=2` before it could be resumed. That made the provider migration fail closed on the one field it was explicitly supposed to transform.

## Fix

The canonical `kaggle/deep_decay_10b_from_15500.py` is now a migration shim in front of the unchanged scientific implementation, preserved as `kaggle/deep_decay_10b_from_15500_impl.py`.

For an existing local/HF deep-decay checkpoint, the shim now:

1. verifies its manifest and exact checkpoint/data cursor;
2. requires the frozen deep-decay WSQD schedule, consumed-token count, scheduler committed-token count, and all schedule parameters to match exactly;
3. accepts only execution microbatch `4` (single-GPU source) or `2` (Kaggle target);
4. for microbatch `4`, rewrites only trainer and scheduler execution microbatch to `2`;
5. recomputes the checkpoint configuration hash using the same model config and staged 10B dataset configuration hash;
6. installs a fresh local manifest without stale publication metadata;
7. immediately runs the original strict deep-decay verifier, which must now accept microbatch `2`, before torchrun can begin.

Model weights, optimizer state, scaler state, RNG state, global step, consumed tokens, data cursor, global optimizer block, and the ADR-0095 learning-rate trajectory are not changed.

Unexpected schedule drift, token drift, cursor drift, or any microbatch other than `4`/`2` still fails closed.

## Commits

- `4d5bc3f9c0d07a1926f5ec4062d111e8927a964b` — preserve the original Kaggle deep-decay implementation under the implementation filename.
- `068d809e72e31bdebe5e54f6d399b394f86e200e` — add the one-time provider execution migration shim.
- `87ccd84c026de39e665ad49de2843cd26cc16772` — add regression contracts for the migration.

No CI status was attached to the regression-test commit at the time of this incident note. The migration shim source was syntax-checked during authoring; the live Kaggle retry remains the decisive hardware/integration gate.

## Retry

The canonical command remains unchanged:

```bash
python kaggle/launch.py deep-decay --model 100M --tokens 10B --max-steps-this-session 250
```

On the retry, an existing downloaded microbatch-4 deep-decay checkpoint should print a `migrated ... execution slicing 4->2` line before the original strict prepare path proceeds. The 250-update segment remains the production throughput/stability qualification gate.
