---
status: current
last_reviewed: 2026-08-21
---

# 100M/10B deep-decay provider switching

The continuation run ID is always:

```text
100m-10b-deep-decay-from-step15500
```

Hugging Face is the provider-neutral checkpoint rendezvous. Do not manually
select a checkpoint when changing providers. The target launcher compares its
local state with the shared HF latest pointer, restores the newest verified
step, validates the frozen scientific state, and applies an execution-only
microbatch/CUDA-RNG topology migration when necessary.

## Launch commands

### Modal H100

```bash
modal run --detach modal/launch.py \
  --action deep-decay \
  --model 100M \
  --tokens 10B
```

Target execution slice: microbatch 16 on one H100.

### Beam RTX4090

```bash
python beam/deep_decay_10b_from_15500.py --gpu RTX4090
```

Target execution slice: microbatch 4 on one GPU. The CPU prepare stage now
pulls a newer HF continuation even when an older checkpoint remains on the Beam
run Volume, then rewrites Kaggle/Modal execution topology to Beam's slice before
allocating the GPU.

### Kaggle 2xT4

```bash
python kaggle/launch.py deep-decay --model 100M --tokens 10B
```

Target execution slice: microbatch 2 with world size 2 and the unchanged global
64-sequence optimizer block. The Kaggle shim accepts checkpoints last written
with microbatch 2, 4, or 16 and canonicalizes CUDA RNG state to two ranks.

## Expected switching behavior

- Modal -> Kaggle: `16/1 RNG state -> 2/2 RNG states`; duplicate rank-zero RNG.
- Beam -> Kaggle: `4/1 -> 2/2`; duplicate rank-zero RNG.
- Kaggle -> Modal: `2/2 -> 16/1`; project rank-zero RNG.
- Kaggle -> Beam: `2/2 -> 4/1`; project rank-zero RNG.
- Beam <-> Modal: one-GPU rank-zero RNG remains byte-exact; only microbatch changes.
- Same-provider resume: no execution rewrite when already canonical.

A provider migration changes floating-point reduction/slicing order, as already
accepted by the provider execution ADRs, but it does not change the scientific
optimizer block or LR trajectory.

## Fail-closed checks

Do not dispatch GPU work if the checkpoint has a malformed/out-of-horizon step,
wrong data cursor, scientific scheduler/LR drift, trainer/scheduler config
mismatch, unauthorized microbatch, invalid CUDA RNG topology, or invalid local
manifest. The original exact step-15,500 source remains a fallback only when no
continuation exists.
