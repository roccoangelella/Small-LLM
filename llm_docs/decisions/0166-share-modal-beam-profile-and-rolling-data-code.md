---
status: accepted
date: 2026-09-08
supersedes: null
owners: [Small-LLM]
---

# ADR 0166: share Modal and Beam profile and rolling-data code

## Context and problem statement

The Modal and Beam adapters independently defined the same model/token registry and carried near-identical rolling-dataset staging and incremental-producer modules. The copies had already drifted only in provider labels, local volume transport names, GPU allowlists, microbatch candidates, and a few runtime details. The new 200M/100B trajectory requires both providers to resolve exactly the same scientific identity, so duplicated registries are an unnecessary drift risk.

The full provider runtime and checkpoint-transport adapters still contain real provider-specific behavior and monkeypatch-compatible module state. Combining those modules in the same change would make the active launch path harder to validate.

## Considered options

- Keep all Modal and Beam copies and rely on symmetry tests.
- Replace every provider module with one large generic runtime immediately.
- Share only the identical profile and rolling-data implementation, while retaining stable provider entrypoints and genuinely provider-specific runtime code.

## Decision outcome

Chosen option: **put the common pretraining profile registry and rolling-data implementation under `providers/`, and keep `modal/` and `beam/` as thin bindings plus SDK-specific launchers.**

The shared profile registry owns model geometry labels, token/dataset mappings, cross-profile restrictions, legacy run IDs, and canonical run naming. Provider bindings continue to own GPU allowlists, default GPU, execution-microbatch candidates, and the finite-dataset volume transport name.

Both provider launchers accept compact `--profile MODEL-TOKENS` selection. Existing `--model MODEL --tokens TOKENS` syntax remains supported, and mixing the two forms fails closed.

`dataset.incremental_cache.IncrementalRollingShardCache` becomes the sole incremental trainer-cache implementation. `dataset.incremental_frontier` retains producer/frontier/staging contracts but no longer carries a second cache class.

Stable operator and remote-import paths remain unchanged:

- `modal/launch.py`, `modal/profiles.py`, `modal/rolling_dataset.py`, and `modal/rolling_producer.py`;
- `beam/launch.py`, `beam/profiles.py`, `beam/rolling_dataset.py`, and `beam/rolling_producer.py`.

The duplicated `modal/runtime.py` and `beam/runtime.py` are deliberately deferred. Their module-level patch points and Beam-only source-migration/Triton behavior need a separate extraction design.

## Consequences

### Positive

- Modal and Beam cannot silently diverge on the 200M/100B scientific profile.
- Rolling frontier staging and production have one implementation while retaining provider-specific logs and bindings.
- The provider directories remain compatible with Modal packaging, Beam source sync, existing runbooks, and deployed import paths.
- Operators can select the active trajectory with one profile argument.

### Negative or limiting

- Thin compatibility bindings remain in both provider directories.
- The large provider runtimes and checkpoint adapters still contain duplication.
- The shared rolling implementation retains compatibility names such as `stage_for_h100` and the existing runtime marker/environment names until a later checkpoint-compatible migration.

## Validation

- Run the complete offline unit suite.
- Assert both provider bindings resolve `200M-100B` to `200m-100b-data-001`, `100b-b64`, and the expanded model preset.
- Assert compact and legacy selector forms resolve identically and mixed syntax fails.
- Run Modal source-packaging and Beam handler/import tests.
- Run incremental cache and durability-order tests.
- Require each provider CPU import preflight to import its stable binding paths before any GPU allocation.

## Links

- `llm_docs/decisions/0061-add-beam-as-an-alternate-single-gpu-training-provider.md`
- `llm_docs/decisions/0158-prebuild-100b-in-public-hf-bucket.md`
- `llm_docs/decisions/0164-restrict-200m-100b-to-modal-and-beam.md`
- `providers/profiles.py`
- `providers/rolling_dataset.py`
- `providers/rolling_producer.py`
