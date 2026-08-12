---
status: superseded
date: 2026-08-11
superseded_by: 0046
---

# 0045 — Run periodic Hugging Face backups only while Modal training is live

## Context and problem statement

ADR 0044 requires the 100M / 2B Modal trajectory to be preserved on Hugging Face in addition to the persistent `small-llm-runs` Modal Volume. A simple ten-minute shell loop around `modal/publish_hf.py` would continue running after the detached training App has stopped, repeatedly re-uploading the same newest verified checkpoint with no durability benefit.

The user wants the periodic Hugging Face backup loop to exist only for the lifetime of the active Modal training run.

## Considered options

- Keep an unconditional ten-minute publisher loop and stop it manually later.
- Poll the Modal App list by name, even though recently stopped Apps remain listed.
- Gate publication on live Modal execution state and automatically terminate the publisher loop when the training App stops.

## Decision outcome

Chosen option: **periodic Hugging Face publication must run only while the `small-llm-training` Modal App is live and must terminate automatically when that training App stops**.

The live-state check must not mutate, restart, or otherwise control the training App. The publication process remains separate from training and reads the persistent run Volume only. Modal execution state, rather than mere presence of a recently stopped App record, is the gate.

This decision was superseded by ADR 0046 after a Modal-account migration exposed that the external publisher is not sufficient as a cross-workspace resume transport and that accumulating distinct checkpoint paths can exhaust private Hub storage.

## Consequences

### Positive

- No repeated all-night publication of an unchanged checkpoint after training has ended.
- The uploader remains operationally independent of the H100 training process.
- Modal Volume remains the exact-resume checkpoint transport while Hugging Face receives periodic off-provider copies during the active run.

### Negative or limiting

- The operator loop depends on Modal CLI observability of the live training execution.
- If the operator session itself dies, periodic Hugging Face backups stop, while Modal Volume checkpoint durability remains unaffected.

## Validation

Start the publisher while `small-llm-training` is live and verify that new verified checkpoints are published at the requested interval. When the training App stops, verify that the publisher loop exits without another periodic upload. The final completed checkpoint can still be explicitly published with the completion-required publication command from ADR 0044.

## Links

- [`0044-publish-100m-2b-final-model-to-hugging-face.md`](0044-publish-100m-2b-final-model-to-hugging-face.md)
- [`0046-use-rolling-hf-as-modal-cross-workspace-checkpoint-transport.md`](0046-use-rolling-hf-as-modal-cross-workspace-checkpoint-transport.md)
- [`../runbooks/modal_training_launcher.md`](../runbooks/modal_training_launcher.md)
- [`../current/status.md`](../current/status.md)
