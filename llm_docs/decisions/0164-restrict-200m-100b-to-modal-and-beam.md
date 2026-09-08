---
status: accepted
date: 2026-09-08
supersedes: null
owners: [Small-LLM]
---

# ADR 0164: restrict the 200M/100B production run to Modal and Beam

## Context and problem statement

The next major pretraining trajectory is approximately 200M parameters / 100B target tokens. The user explicitly restricted this run to the two production cloud lanes already used for long pretraining work: Modal and Beam.

The current 100B launchers already enforce provider-specific GPU lanes: Modal H100 and Beam RTX4090. Kaggle is not operationally appropriate for this horizon and is not part of the requested run surface.

## Decision outcome

Chosen option: **wire and support the 200M/100B production run only on Modal and Beam.**

For this run, preserve the existing 100B provider GPU restrictions unless separately superseded:

- Modal: H100 lane;
- Beam: RTX4090 lane;
- Kaggle: no 200M/100B production launcher.

Provider differences may change only execution slicing, staging, retry behavior, and provider-specific durability mechanics. They must not change model geometry, optimizer recipe, LR schedule, dataset order, global optimizer block, or checkpoint scientific identity.

## Consequences

### Positive

- The run uses the two provider paths already designed for long single-GPU pretraining and cross-provider resume.
- Scientific configuration remains provider-neutral while execution details stay adapter-specific.
- No time is spent qualifying Kaggle for a trajectory that is operationally unsuitable there.

### Negative or limiting

- The 200M geometry still requires fresh memory/throughput qualification on both selected GPUs.
- Provider-specific microbatch sizes may differ and must be measured rather than assumed.

## Validation

Before production launch:

1. both provider profile tables must resolve `200M` / `100B` to the same scientific run identity;
2. Modal H100 and Beam RTX4090 must pass import/data-stage/microbatch/training smoke gates;
3. checkpoint identity and exact resume must match across providers;
4. no Kaggle 200M/100B production entrypoint may be added as part of this wiring.

## Links

- `llm_docs/decisions/0160-select-200m-100b-as-next-pretraining-run.md`
- `llm_docs/decisions/0162-scale-100m-10b-lr-phase-anchors-proportionally-to-200m-100b.md`
- `llm_docs/decisions/0163-freeze-200m-100b-lr-anchors.md`
- `llm_docs/decisions/0158-prebuild-100b-in-public-hf-bucket.md`
