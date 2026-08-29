---
status: accepted
date: 2026-08-21
supersedes: provider-lock portions of 0099 and 0114
---

# 0117 — Make the 100M/10B deep-decay continuation portable across Modal, Beam, and Kaggle

## Context

ADR 0095 freezes the scientific 100M/10B deep-decay trajectory under the run ID
`100m-10b-deep-decay-from-step15500`. Later execution ADRs moved that same
trajectory between Beam, Kaggle two-T4 DDP, and Modal H100 while preserving the
64-sequence optimizer block.

The user expects to switch providers frequently according to available credits
and capacity. Provider switching must therefore be a normal resume operation,
not a bespoke migration each time.

## Decision

The shared Hugging Face model-repository namespace is the durable rendezvous for
all three execution providers. Before allocating training GPUs, each provider
must resolve the newest manifest-verified continuation checkpoint available
from its local durable/scratch state and Hugging Face, preferring the greater
valid step.

Provider-specific execution geometry is canonicalized automatically:

| provider | GPU topology | execution microbatch | CUDA RNG states |
|---|---:|---:|---:|
| Modal | 1x H100 | 16 | 1 |
| Beam | 1x RTX4090/compatible single GPU | 4 | 1 |
| Kaggle | 2x T4 DDP | 2 | 2 |

Authorized migrations may change only:

- `config.microbatch_size`;
- `scheduler.config.microbatch_size` to keep scheduler/trainer config identity;
- the derived checkpoint configuration hash;
- CUDA RNG topology cardinality required by the target provider.

For two GPUs to one GPU, retain the byte-exact rank-zero CUDA RNG state. For one
GPU to two GPUs, deterministically duplicate the byte-exact rank-zero CUDA RNG
state to both target ranks. Existing canonical two-rank RNG states are retained
on Kaggle same-provider resume.

Model weights, optimizer state, scaler, Python RNG, CPU Torch RNG, scientific
scheduler fields and committed LR, global step, consumed tokens, data cursor,
64-sequence optimizer block, validation prefix, dataset ordering, precision,
and ADR-0095 schedule are invariant and must fail closed on drift.

The provider adapter may keep a local pre-migration backup while rewriting a
restored checkpoint. The rewritten resume checkpoint need not replace the
provider-neutral HF source checkpoint: the next normal training checkpoint is
published under the unchanged shared run namespace.

## Operational consequence

Switching provider is now intended to require only that provider's normal
launch command. No checkpoint ID or manual `--resume` argument is supplied.

```bash
# Modal H100
modal run --detach modal/launch.py --action deep-decay --model 100M --tokens 10B

# Beam RTX4090
python beam/deep_decay_10b_from_15500.py --gpu RTX4090

# Kaggle 2xT4
python kaggle/launch.py deep-decay --model 100M --tokens 10B
```

Each command must discover the newest verified continuation itself, migrate its
execution topology if necessary, and resume the same scientific trajectory.

## Validation

A provider-neutral migration helper must cover `16 -> 2`, `4 -> 2`, `2 -> 4`,
`2 -> 16`, `4 -> 16`, and same-provider no-op/canonicalization cases. It must
reject unauthorized microbatches, scheduler/trainer config disagreement, and
invalid CUDA RNG cardinality. Provider adapters retain their stronger
scientific checkpoint and data-cursor checks before GPU dispatch.

## Links

- [`0095-decay-1e-4-to-1e-5-then-5e-6.md`](0095-decay-1e-4-to-1e-5-then-5e-6.md)
- [`0099-run-deep-decay-100m-10b-on-kaggle-dual-t4.md`](0099-run-deep-decay-100m-10b-on-kaggle-dual-t4.md)
- [`0114-run-deep-decay-100m-10b-on-modal-h100.md`](0114-run-deep-decay-100m-10b-on-modal-h100.md)
- [`../runbooks/100m_10b_deep_decay_provider_switching.md`](../runbooks/100m_10b_deep_decay_provider_switching.md)
- [`../../trainer/deep_decay_provider_migration.py`](../../trainer/deep_decay_provider_migration.py)
