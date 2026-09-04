# Decision log

Each ADR records one durable choice, its context, alternatives, outcome, and consequences. Accepted ADRs are not rewritten to hide changed reasoning; create a new ADR that supersedes the old one. Use [`template.md`](template.md) for new decisions.

## Status meanings

- `accepted`: current durable decision.
- `superseded`: replaced by a later ADR.
- `proposed`: under discussion, not authorized.
- `rejected`: considered and explicitly not chosen.

## Accepted ADRs by Domain

### 1. Architecture & CUDA Kernels (FLA / GDN-2)
- [`0003-defer-architecture-baselines-until-larger-models.md`](0003-defer-architecture-baselines-until-larger-models.md)
- [`0005-adapt-gdn2-chunks-to-decay-span.md`](0005-adapt-gdn2-chunks-to-decay-span.md)
- [`0006-calibrate-fp16-loss-scale-before-failing-block.md`](0006-calibrate-fp16-loss-scale-before-failing-block.md)
- [`0011-publish-standalone-gated-delta-rule-package.md`](0011-publish-standalone-gated-delta-rule-package.md)
- [`0013-rename-public-gdr2-repository.md`](0013-rename-public-gdr2-repository.md)
- [`0014-simplify-public-gdr2-repository-docs.md`](0014-simplify-public-gdr2-repository-docs.md)
- [`0015-scrub-small-llm-references-from-public-gdr2.md`](0015-scrub-small-llm-references-from-public-gdr2.md)
- [`0016-qualify-fla-gdn2-before-changing-decay.md`](0016-qualify-fla-gdn2-before-changing-decay.md)
- [`0018-integrate-fla-gdn2-as-checkpoint-compatible-cuda-backend.md`](0018-integrate-fla-gdn2-as-checkpoint-compatible-cuda-backend.md)
- [`0019-resume-500m-checkpoint-with-fla-gdn2-execution.md`](0019-resume-500m-checkpoint-with-fla-gdn2-execution.md)
- [`0020-qualify-fla-gdn2-with-full-fp32-kernel-execution.md`](0020-qualify-fla-gdn2-with-full-fp32-kernel-execution.md)
- [`0021-qualify-fla-gdn2-v052-and-resume-step4000.md`](0021-qualify-fla-gdn2-v052-and-resume-step4000.md)
- [`0035-retire-completed-fla-investigation-scripts-from-kaggle.md`](0035-retire-completed-fla-investigation-scripts-from-kaggle.md)

### 2. Pretraining & Infrastructure (Modal, Beam, Kaggle)
- [`0004-run-100m-in-one-session-with-250-step-durability.md`](0004-run-100m-in-one-session-with-250-step-durability.md)
- [`0009-start-500m-at-microbatch-4-with-250-step-durability.md`](0009-start-500m-at-microbatch-4-with-250-step-durability.md)
- [`0023-run-2b-20m-probe-via-vps-kaggle-dataset.md`](0023-run-2b-20m-probe-via-vps-kaggle-dataset.md)
- [`0027-use-500m-schema-gains-to-justify-fixed-20m-token-scaling-through-2b.md`](0027-use-500m-schema-gains-to-justify-fixed-20m-token-scaling-through-2b.md)
- [`0028-use-one-profile-driven-launcher-for-publication-and-training.md`](0028-use-one-profile-driven-launcher-for-publication-and-training.md)
- [`0030-consolidate-kaggle-profile-wrappers-behind-one-runtime.md`](0030-consolidate-kaggle-profile-wrappers-behind-one-runtime.md)
- [`0039-use-modal-for-future-gpu-training.md`](0039-use-modal-for-future-gpu-training.md)
- [`0041-use-block64-modal-corpus-and-probe-microbatch-16-32-48-64.md`](0041-use-block64-modal-corpus-and-probe-microbatch-16-32-48-64.md)
- [`0044-publish-100m-2b-final-model-to-hugging-face.md`](0044-publish-100m-2b-final-model-to-hugging-face.md)
- [`0049-make-modal-launch-logs-concise-and-unambiguous.md`](0049-make-modal-launch-logs-concise-and-unambiguous.md)
- [`0051-qualify-exact-batch-dual-t4-ddp-before-kaggle-adoption.md`](0051-qualify-exact-batch-dual-t4-ddp-before-kaggle-adoption.md)
- [`0056-adopt-exact-batch-dual-t4-ddp-for-kaggle-only.md`](0056-adopt-exact-batch-dual-t4-ddp-for-kaggle-only.md)
- [`0057-use-standard-wsd-for-100m-10b.md`](0057-use-standard-wsd-for-100m-10b.md)
- [`0060-require-live-modal-hf-smoke-before-100m-10b.md`](0060-require-live-modal-hf-smoke-before-100m-10b.md)
- [`0061-add-beam-as-an-alternate-single-gpu-training-provider.md`](0061-add-beam-as-an-alternate-single-gpu-training-provider.md)
- [`0062-default-beam-training-to-serverless-rtx5090.md`](0062-default-beam-training-to-serverless-rtx5090.md)
- [`0065-cap-beam-startup-probe-at-microbatch-16.md`](0065-cap-beam-startup-probe-at-microbatch-16.md)
- [`0069-own-beam-launch-namespace-in-synced-checkout.md`](0069-own-beam-launch-namespace-in-synced-checkout.md)
- [`0071-run-full-100m-10b-with-concurrent-5b-evaluation.md`](0071-run-full-100m-10b-with-concurrent-5b-evaluation.md)
- [`0072-pin-beam-client-0207-for-live-gateway.md`](0072-pin-beam-client-0207-for-live-gateway.md)
- [`0114-run-deep-decay-100m-10b-on-modal-h100.md`](0114-run-deep-decay-100m-10b-on-modal-h100.md)
- [`0132-split-latest-checkpoints-to-hf-bucket-and-best-model-to-dedicated-repo.md`](0132-split-latest-checkpoints-to-hf-bucket-and-best-model-to-dedicated-repo.md)
- [`0144-unify-100m-10b-pretraining-probes.md`](0144-unify-100m-10b-pretraining-probes.md)

### 3. Datasets, Tokenization & Remote Durability
- [`0043-prepare-modal-block64-corpus-on-vps.md`](0043-prepare-modal-block64-corpus-on-vps.md)
- [`0048-verify-modal-dataset-in-active-workspace.md`](0048-verify-modal-dataset-in-active-workspace.md)
- [`0053-stream-10b-through-one-gib-hf-shards-and-cpu-stage-before-h100.md`](0053-stream-10b-through-one-gib-hf-shards-and-cpu-stage-before-h100.md)
- [`0054-retire-google-drive-for-new-dataset-durability.md`](0054-retire-google-drive-for-new-dataset-durability.md)
- [`0058-produce-10b-shards-concurrently-with-modal-training.md`](0058-produce-10b-shards-concurrently-with-modal-training.md)
- [`0070-use-vps-fed-beam-volume-for-10b-dataset-production.md`](0070-use-vps-fed-beam-volume-for-10b-dataset-production.md)

### 4. Post-Training & R-SFT Reasoning
- [`0032-scale-sft-budget-with-pretraining-and-qualify-on-500m-first.md`](0032-scale-sft-budget-with-pretraining-and-qualify-on-500m-first.md)
- [`0033-use-comprehensive-post-sft-qualification-and-pretraining-cadence.md`](0033-use-comprehensive-post-sft-qualification-and-pretraining-cadence.md)
- [`0034-make-sft-data-publication-machine-agnostic.md`](0034-make-sft-data-publication-machine-agnostic.md)
- [`0038-stream-local-chat-tokens-as-generated.md`](0038-stream-local-chat-tokens-as-generated.md)
- [`0063-document-local-chat-cli-usage-in-source.md`](0063-document-local-chat-cli-usage-in-source.md)
- [`0064-allow-stable-pretrained-artifacts-in-local-chat.md`](0064-allow-stable-pretrained-artifacts-in-local-chat.md)
- [`0067-keep-100m-2b-sft-at-4-percent-on-dual-t4.md`](0067-keep-100m-2b-sft-at-4-percent-on-dual-t4.md)
- [`0073-persist-sft-cadence-checkpoints-before-evaluation.md`](0073-persist-sft-cadence-checkpoints-before-evaluation.md)
- [`0074-stream-ddp-sft-checkpoints-with-bounded-host-memory.md`](0074-stream-ddp-sft-checkpoints-with-bounded-host-memory.md)
- [`0075-bound-dual-t4-sft-inline-qualification.md`](0075-bound-dual-t4-sft-inline-qualification.md)
- [`0076-bound-continuous-kaggle-sft-host-memory.md`](0076-bound-continuous-kaggle-sft-host-memory.md)
- [`0077-start-reasoning-sft-with-three-shuffled-concise-difficulty-bands.md`](0077-start-reasoning-sft-with-three-shuffled-concise-difficulty-bands.md)
- [`0078-define-reasoning-skill-contract-before-r-sft-data.md`](0078-define-reasoning-skill-contract-before-r-sft-data.md)
- [`0079-use-special-reasoning-tokens-and-defer-adaptive-thinking-policy-to-rl.md`](0079-use-special-reasoning-tokens-and-defer-adaptive-thinking-policy-to-rl.md)
- [`0080-use-gemini-api-as-rsft-teacher-despite-contract-caveat.md`](0080-use-gemini-api-as-rsft-teacher-despite-contract-caveat.md)
- [`0081-track-reasoning-difficulty-labels-for-telemetry-only.md`](0081-track-reasoning-difficulty-labels-for-telemetry-only.md)
- [`0082-focus-r0-on-logic-primitives-and-defer-exact-computation.md`](0082-focus-r0-on-logic-primitives-and-defer-exact-computation.md)
- [`0083-wire-rsft-gemrouter-transport-before-prompt-policy.md`](0083-wire-rsft-gemrouter-transport-before-prompt-policy.md)
- [`0106-resume-expanded-rsft-with-curation-v2-and-keeper-only-gemini.md`](0106-resume-expanded-rsft-with-curation-v2-and-keeper-only-gemini.md)
- [`0108-promote-expanded-rsft-corpus-to-kaggle-default.md`](0108-promote-expanded-rsft-corpus-to-kaggle-default.md)
- [`0109-bind-100m-rsft-to-profile-specific-hf-repository.md`](0109-bind-100m-rsft-to-profile-specific-hf-repository.md)
- [`0110-disable-xet-for-kaggle-rsft-checkpoint-upload.md`](0110-disable-xet-for-kaggle-rsft-checkpoint-upload.md)
- [`0111-allow-production-rsft-epoch-count.md`](0111-allow-production-rsft-epoch-count.md)
- [`0112-scale-rsft-with-nested-1-2-4-percent-superior-reasoning-corpora.md`](0112-scale-rsft-with-nested-1-2-4-percent-superior-reasoning-corpora.md)
- [`0113-use-superior-reasoning-stage2-for-rsft-scaling.md`](0113-use-superior-reasoning-stage2-for-rsft-scaling.md)
- [`0115-refactor-rsft-dataset-production-into-source-adapters-generic-context-repair-and-main-builder.md`](0115-refactor-rsft-dataset-production-into-source-adapters-generic-context-repair-and-main-builder.md)
- [`0117-reject-expanded-e3-rsft-as-qualified-default.md`](0117-reject-expanded-e3-rsft-as-qualified-default.md)
- [`0119-publish-frozen-rsft-1pct-corpus-on-main.md`](0119-publish-frozen-rsft-1pct-corpus-on-main.md)
- [`0120-ignore-obsolete-rsft-datasets-from-git.md`](0120-ignore-obsolete-rsft-datasets-from-git.md)
- [`0131-use-rolling-latest-only-retention-for-kaggle-sft.md`](0131-use-rolling-latest-only-retention-for-kaggle-sft.md)
- [`0138-start-100m-10b-sft-pipeline-wiring.md`](0138-start-100m-10b-sft-pipeline-wiring.md)
- [`0139-run-100m-10b-sft-on-2b-10pct-s0-data.md`](0139-run-100m-10b-sft-on-2b-10pct-s0-data.md)

### 5. Evaluation, Memory Governance & Repository Tooling
- [`0001-use-structured-markdown-project-memory.md`](0001-use-structured-markdown-project-memory.md)
- [`0002-freeze-eval-core-v1-and-unified-cli.md`](0002-freeze-eval-core-v1-and-unified-cli.md)
- [`0007-render-teacher-forced-examples-as-readable-ground-truth.md`](0007-render-teacher-forced-examples-as-readable-ground-truth.md)
- [`0010-self-provision-eval-core-from-main-evaluator.md`](0010-self-provision-eval-core-from-main-evaluator.md)
- [`0017-use-latest-pointer-for-10m-prompt-comparison.md`](0017-use-latest-pointer-for-10m-prompt-comparison.md)
- [`0025-freeze-canonical-full-post-pretraining-prompt-suite.md`](0025-freeze-canonical-full-post-pretraining-prompt-suite.md)
- [`0031-govern-project-memory-with-progressive-disclosure.md`](0031-govern-project-memory-with-progressive-disclosure.md)
- [`0037-consolidate-dataset-profile-tools-and-retire-one-off-qualification-code.md`](0037-consolidate-dataset-profile-tools-and-retire-one-off-qualification-code.md)
- [`0059-run-supplementary-sampled-three-way-full-evaluation.md`](0059-run-supplementary-sampled-three-way-full-evaluation.md)
- [`0140-define-evaluation-v2-protocol.md`](0140-define-evaluation-v2-protocol.md)
- [`0141-activate-evaluation-v2-entrypoints.md`](0141-activate-evaluation-v2-entrypoints.md)
- [`0142-ignore-local-agent-tooling-directories.md`](0142-ignore-local-agent-tooling-directories.md)
- [`0143-remove-ire-from-project-infrastructure.md`](0143-remove-ire-from-project-infrastructure.md)
- [`0145-synchronize-readme-lifecycle-with-current-state.md`](0145-synchronize-readme-lifecycle-with-current-state.md)

## Superseded ADRs

- [`0055-unify-modal-checkpoints-on-hf-model-repository.md`](0055-unify-modal-checkpoints-on-hf-model-repository.md) — superseded by ADR 0132 (rolling checkpoints in mutable HF Storage Bucket, strict val-loss best in dedicated model repo).
- [`0116-promote-expanded-e3-rsft-as-current-default-r0.md`](0116-promote-expanded-e3-rsft-as-current-default-r0.md) — superseded by ADR 0117 (e3 full qualification showed benchmark regression).
- [`0099-run-deep-decay-100m-10b-on-kaggle-dual-t4.md`](0099-run-deep-decay-100m-10b-on-kaggle-dual-t4.md) — superseded by ADR 0114 (Modal H100 execution with microbatch-16 slicing).
- [`0050-scale-100m-to-fresh-10b-with-5b-capability-gate.md`](0050-scale-100m-to-fresh-10b-with-5b-capability-gate.md) — superseded by ADR 0071 (authorized full 10B with concurrent non-blocking 5B eval).
- [`0068-make-plain-uv-sync-install-complete-runtime.md`](0068-make-plain-uv-sync-install-complete-runtime.md) — complete runtime retained; ADR 0072 pins Beam client 0.2.207.
- [`0066-run-100m-2b-sft-at-10-percent-on-dual-t4.md`](0066-run-100m-2b-sft-at-10-percent-on-dual-t4.md) — superseded by ADR 0067 (restored 4% parent SFT budget).
- [`0008-run-500m-final-20m-data-scaling-probe.md`](0008-run-500m-final-20m-data-scaling-probe.md) — superseded by later 20M scaling choices (ADR 0023).
- [`0012-bind-public-gdr2-repository-identity.md`](0012-bind-public-gdr2-repository-identity.md) — superseded by ADR 0013 (repository renamed).
- [`0022-run-1b-20m-probe-via-vps-kaggle-dataset.md`](0022-run-1b-20m-probe-via-vps-kaggle-dataset.md) — superseded by ADR 0023 (changed target to 2B tokens).
- [`0024-freeze-canonical-questions-only-prompt-test-settings.md`](0024-freeze-canonical-questions-only-prompt-test-settings.md) — superseded by ADR 0025 (canonical full qualitative prompt suite).
- [`0026-prune-superseded-one-off-kaggle-diagnostics.md`](0026-prune-superseded-one-off-kaggle-diagnostics.md) — superseded by ADR 0035 (FLA diagnostic retirements).
- [`0029-limit-pre-2b-kaggle-cleanup-to-dead-wrappers-and-dispatch-fixes.md`](0029-limit-pre-2b-kaggle-cleanup-to-dead-wrappers-and-dispatch-fixes.md) — superseded by ADR 0030 (consolidated profile wrappers).
- [`0036-add-local-completed-sft-chat-cli.md`](0036-add-local-completed-sft-chat-cli.md) — superseded by ADR 0064 (expanded CLI to stable pretrained artifacts).
- [`0040-launch-100m-2b-pretraining-on-modal.md`](0040-launch-100m-2b-pretraining-on-modal.md) — superseded by ADR 0041 (byte-preserving block 64).
- [`0042-derive-modal-block64-corpus-on-kaggle.md`](0042-derive-modal-block64-corpus-on-kaggle.md) — superseded by ADR 0043 (VPS dataset build).
- [`0045-run-periodic-hf-backups-only-while-modal-training-is-live.md`](0045-run-periodic-hf-backups-only-while-modal-training-is-live.md) — superseded by ADR 0046 (integrated HF publication).
- [`0046-use-rolling-hf-as-modal-cross-workspace-checkpoint-transport.md`](0046-use-rolling-hf-as-modal-cross-workspace-checkpoints.md) — superseded by ADR 0047 (HF Storage Bucket).
- [`0047-use-hf-storage-bucket-for-modal-cross-workspace-checkpoints.md`](0047-use-hf-storage-bucket-for-modal-cross-workspace-checkpoints.md) — superseded by ADR 0055 (unified model repo).
- [`0052-evaluate-modal-rolling-checkpoints-directly-from-hf-bucket.md`](0052-evaluate-modal-rolling-checkpoints-directly-from-hf-bucket.md) — superseded by ADR 0055.

Historical omnibus registers live under [`../archive/decision_registers/`](../archive/decision_registers/decisions_and_ablations.md).