---
status: accepted
date: 2026-08-11
supersedes: null
---

# 0044 — Publish the 100M / 2B final model to Hugging Face

## Context and problem statement

The live approximately-100M-parameter / 2B-token Modal run durably checkpoints every 250 optimizer updates to the persistent `small-llm-runs` Modal Volume. The current Modal launcher deliberately disables the trainer's legacy dataset-keyed Hugging Face checkpoint publication because that namespace can collide when the same finite corpus is reused by different model sizes. As a result, the current live path does not itself publish the trained 100M artifact to Hugging Face.

The user requires the completed 100M / 2B model to be preserved on Hugging Face as well as in Modal durable storage.

## Considered options

- Keep Modal Volume as the only durable model/checkpoint store.
- Re-enable the existing dataset-keyed periodic Hugging Face checkpoint protocol unchanged.
- Keep Modal Volume as the live checkpoint transport, but publish the completed verified 100M / 2B artifact to Hugging Face under a model/run-specific identity.

## Decision outcome

Chosen option: **keep Modal Volume checkpointing for the live run and additionally publish the completed verified 100M / 2B model artifact to Hugging Face under a model/run-specific identity**.

The already-running training trajectory must not be restarted or scientifically mutated merely to add Hugging Face publication. The publication path must be able to operate from the verified durable Modal checkpoint after completion. At minimum, the final verified checkpoint/model weights and the metadata needed to identify the run, model geometry, dataset, source commit, and training step must be present on Hugging Face.

## Consequences

### Positive

- The 100M / 2B trained model is not stranded only in Modal storage.
- The live run retains its existing 250-step Modal durability and exact-resume behavior.
- Model/run-specific Hugging Face identity avoids the known cross-model collision in the legacy dataset-keyed namespace.
- Publication can be added without perturbing the active frozen training trajectory.

### Negative or limiting

- A dedicated final-export/publication path is required; simply setting `--remote-publish-every-steps` on the current Modal run is not sufficient because the legacy protocol is dataset-keyed and expects Drive-manifest evidence.
- Until that publication path is implemented and executed successfully, the Hugging Face copy is not yet guaranteed to exist.

## Validation

This decision is satisfied when the final verified checkpoint of W&B run `100m-2b-data-001` can be resolved from `small-llm-runs`, published to the configured private Hugging Face repository under a model/run-specific namespace, downloaded into a clean environment, and verified to match the source checkpoint identity and model state.

## Links

- [`../current/status.md`](../current/status.md)
- [`../runbooks/modal_training_launcher.md`](../runbooks/modal_training_launcher.md)
- [`0041-use-block64-modal-corpus-and-probe-microbatch-16-32-48-64.md`](0041-use-block64-modal-corpus-and-probe-microbatch-16-32-48-64.md)
- [`0043-prepare-modal-block64-corpus-on-vps.md`](0043-prepare-modal-block64-corpus-on-vps.md)
