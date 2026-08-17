---
status: accepted
date: 2026-08-17
supersedes: 0071
---

# 0090 — Pause the full 100M / 10B trajectory for a step-12,500 matched-decay probe

## Context

The live 100M/10B validation curve stays in the long WSD stable phase while the completed 100M/2B endpoint received its terminal cooldown around the point where the two curves begin to separate. The earlier 20M/500M → 20M/2B experiment showed the same visual pattern: a long stable-LR plateau followed by a large cooldown loss drop, while the final extra-data gain remained modest. This makes spending the remainder of the 10B horizon before measuring the latent benefit unnecessarily expensive.

The user selected step 12,500 of `100m-10b-data-001` as the counterfactual fork point.

## Decision

Pause/terminate the uncapped full 100M/10B Beam trajectory once the exact local Beam Volume checkpoint `step-00012500` is confirmed durable. Preserve the original run, dataset, and checkpoints; do not delete or rewrite them.

Run a temporary diagnostic fork named `100m-10b-decay-probe-step12500` from that exact checkpoint. The fork must preserve model parameters, optimizer state, scaler state, RNG state, and data cursor. The only scientific change is the LR schedule.

Use the exact historical `modal-2b-b64` WSD schedule as the controlled counterfactual:

```text
peak LR:          3e-4
warmup tokens:    100,007,936
stable tokens:  1,499,987,968
decay tokens:     399,998,976
minimum LR ratio: 0.1
```

At 12,500 block-64 updates the 10B checkpoint has consumed 1,638,400,000 targets, so this matched schedule is already about 9.60% through its cooldown. The fork therefore starts at approximately `2.939e-4`, not at an arbitrary fixed low LR, and cosine-decays to `3e-5`.

Run only through the historical 2B schedule endpoint. That requires 2,759 additional full block-64 updates, approximately 361.63M additional targets, ending at global step 15,259. Use the 10B corpus at the checkpoint's exact next block, frozen 16-block validation prefix, microbatch 4, FP16, and the unchanged hybrid Muon+AdamW recipe.

The temporary launcher is `beam/decay_probe_12500.py`. It must fail closed if the exact step-12,500 source checkpoint is unavailable; it may not silently substitute latest or nearest state. CPU staging must prepare and verify the checkpoint-aligned dataset window before allocating the GPU. Probe W&B/checkpoint publication uses a separate run ID and must not mutate the original 10B run namespace.

## Consequences

- ADR 0071's instruction to keep the 100M/10B run uncapped through all 76,294 updates is superseded.
- This probe answers a narrower question before further 10B compute is authorized: whether the step-12,500 10B state already contains useful learning that is hidden by the stable LR.
- The probe is not a new production pretraining recipe and does not redefine the standard WSD contract globally.
- A later decision is required to resume/replace the full 10B trajectory after the matched-decay result is evaluated against the completed 100M/2B endpoint.

## Links

- [`0057-use-standard-wsd-for-100m-10b.md`](0057-use-standard-wsd-for-100m-10b.md)
- [`0071-run-full-100m-10b-with-concurrent-5b-evaluation.md`](0071-run-full-100m-10b-with-concurrent-5b-evaluation.md)
- [`../reference/100m_10b_incremental_dataset.md`](../reference/100m_10b_incremental_dataset.md)
- [`../current/roadmap.md`](../current/roadmap.md)
