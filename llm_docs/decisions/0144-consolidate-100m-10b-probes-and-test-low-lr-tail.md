---
status: accepted
date: 2026-09-04
supersedes: 0135
owners: [Small-LLM]
---

# ADR 0144: consolidate 100M/10B probes and test the low-LR tail

## Context

The completed 100M/10B pretraining run entered a terminal learning-rate decay near the end of training. Its late validation-loss curve therefore cannot by itself distinguish model/data saturation from an optimizer that has simply been driven to a very small step size.

The older Probe A experiment used separate `probe_a_lr_reset_10b.py` and `probe_a_lr_reset_10b_impl.py` launch files and tested large LR resets. The historical W&B run `100m-10b-probe-a-reset-low-from-step71750` is retained as evidence. The 1e-4 late reset was harmful, so the new active question is not whether to reheat aggressively but whether holding a small non-decaying LR allows validation loss to keep falling.

The historical step-00071750 source may no longer be retained in Hugging Face because the dedicated best-model repository is replace-oriented. The current strict-best artifact may therefore be the only usable source.

## Decision

`kaggle/src/probes_100m_10b.py` is the single public home for short 100M/10B pretraining probes. Do not add a new one-off Kaggle Python file for each future 100M/10B probe; extend this launcher instead.

The active paired probes are:

- `hold-1e-5`: constant LR `1e-5` for 3,000 updates;
- `hold-2e-5`: constant LR `2e-5` for 3,000 updates.

Both branches must use the same source checkpoint and identical following corpus blocks. Validation runs every 250 updates over the frozen 16-block validation set. Checkpoints remain local-only and Hugging Face publication is disabled.

The source policy is:

1. prefer the exact historical `step-00071750` checkpoint from the dedicated 100M/10B best-model repository if it is still physically available;
2. if and only if that artifact is absent, read `best_model.json` and use the strict-best checkpoint it currently names;
3. never silently fall back to rolling `latest`;
4. encode the actual selected source step in every W&B probe run ID.

The old constant-`1e-4` reset remains available only as a legacy reproduction target. The old `3e-4` reset branch is retired from the active probe set.

For this experiment stage, the primary decision criterion is validation loss: **if a controlled branch produces a sustained lower validation loss on the same frozen validation set, treat that checkpoint as a better pretrained model for the purpose of deciding whether the 100M model still benefits from additional optimization/data.** Downstream benchmark and qualitative evaluation can still be used later to characterize what the lower loss buys behaviorally, but they are not required to acknowledge the intrinsic improvement.

## Scope limitation

The current deterministic corpus ends at the 10B endpoint. These probes therefore test the remaining tail from the chosen pre-terminal source checkpoint under a held LR; they do not constitute a true >10B fresh-data continuation. A 20B/50B/100B continuation requires extending the deterministic corpus first.

## Repository consequences

- Canonical launcher: `kaggle/src/probes_100m_10b.py`.
- Superseded launcher files `kaggle/src/probe_a_lr_reset_10b.py` and `kaggle/src/probe_a_lr_reset_10b_impl.py` are removed.
- Static probe contracts target the canonical launcher.
- `kaggle/src/README.md` instructs future work to add 100M/10B probes to the canonical file rather than creating new one-off launchers.

## Operational examples

```bash
python kaggle/src/probes_100m_10b.py --dry-run
python kaggle/src/probes_100m_10b.py --probe active
python kaggle/src/probes_100m_10b.py --probe hold-1e-5
python kaggle/src/probes_100m_10b.py --probe hold-2e-5
```

Before allocating GPUs, the non-dry-run path must print the selected source checkpoint and whether it came from the preferred historical step or the current strict-best fallback.
