---
status: accepted
date: 2026-09-11
supersedes: null
---

# Publish the MoE 100B SuperBPE corpus in a public HF bucket

## Context

The frozen MoE corpus entrypoint already reuses the established ClimbMix
selection, cluster stratification, deterministic split, schema-v2 sharding, and
incremental READY protocol while retokenizing accepted documents with the
frozen 8k SuperBPE tokenizer. Its HF Storage Bucket transport was still
hardcoded private, which did not satisfy the publication requirement.

The existing shared dataset bucket is private and contains earlier corpora, so
changing its visibility would expose unrelated artifacts.

## Decision

Publish `moe-100b-superbpe-b64-dataset-001` to a dedicated public HF Storage
Bucket. The frozen entrypoint forces public visibility, the storage adapter
verifies observed visibility after bucket creation, and the Modal launcher
accepts the dedicated bucket ID explicitly instead of relying on the shared
private default.

The tokenizer artifact, stratification weights, source revision, selection
seed, target horizon, and schema remain unchanged.

## Consequences

- Production fails closed if the destination is not public.
- Earlier private datasets remain private.
- Public visibility is part of the emitted transport metadata and test contract.
- The long-running Modal producer remains resumable through the existing volume
  and incremental frontier.
