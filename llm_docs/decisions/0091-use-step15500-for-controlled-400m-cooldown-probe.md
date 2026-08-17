---
status: accepted
date: 2026-08-17
supersedes: 0090
---

# 0091 — Use step 15,500 for the controlled 400M cooldown probe

## Context

The originally selected `step-00012500` snapshot for `100m-10b-data-001` is no longer recoverable. A read-only Beam/Hugging Face inspection found that the Beam run Volume now starts at step 15,500, while the rolling latest-only Hugging Face model-repository tree retains only step 23,500. Therefore step 15,500 is the earliest exact recoverable state and the closest available state to the originally intended divergence point.

Step 15,500 has consumed 2,031,616,000 target tokens. Reusing the historical 100M/2B WSD schedule in absolute-token coordinates would already place this state beyond that schedule's cooldown endpoint, so it would collapse immediately to the minimum LR rather than test a controlled anneal.

## Decision

Supersede ADR 0090's step-12,500 matched-absolute-schedule probe.

Fork the exact local Beam checkpoint `step-00015500` into a separate diagnostic run `100m-10b-decay-probe-step15500`. The probe must fail closed if that exact checkpoint is unavailable and must never substitute a later or nearest checkpoint.

Preserve model parameters, optimizer state, scaler state, RNG state, data cursor, model architecture, optimizer recipe, FP16 precision, microbatch 4, frozen validation prefix, and exact 10B corpus order. Change only the LR scheduler.

Use a controlled cosine cooldown that starts at the production peak LR at the fork point and decays to the existing minimum LR ratio over approximately 400M target tokens:

```text
source step:               15,500
source consumed targets:   2,031,616,000
peak/start LR:              3e-4
minimum LR ratio:           0.1
final LR:                   3e-5
requested cooldown targets: 400,000,000
block-aligned decay span:   400,031,744 targets
cooldown updates:           3,052
final global step:          18,552
```

Implement this with WSD in absolute committed-token space using `warmup_tokens=0`, `stable_tokens=2,031,616,000`, and `decay_tokens=400,031,744`. Thus LR is exactly `3e-4` at the source checkpoint and the next successful optimizer update begins cosine decay.

Use `beam/decay_probe_15500.py` as the temporary launcher. The old `beam/decay_probe_12500.py` is retained only as a compatibility redirect so stale operator commands cannot launch the obsolete experiment. CPU-stage and verify the checkpoint-aligned dataset window before GPU allocation. Keep W&B and Hugging Face checkpoint publication in the separate probe run namespace and use the project's current HF model-repository checkpoint transport.

## Consequences

- Step 15,500 is now the only authorized fork point for this diagnostic.
- The experiment measures whether the earliest still-recoverable stable-LR 10B state benefits from a conventional approximately-400M-token cooldown.
- This is a diagnostic branch, not a new global pretraining schedule.
- Further 10B training remains gated on comparing the cooled probe against the completed 100M/2B endpoint.

## Links

- [`0090-pause-full-100m-10b-for-step12500-matched-decay-probe.md`](0090-pause-full-100m-10b-for-step12500-matched-decay-probe.md)
- [`0057-use-standard-wsd-for-100m-10b.md`](0057-use-standard-wsd-for-100m-10b.md)
- [`../runbooks/100m_10b_beam.md`](../runbooks/100m_10b_beam.md)
