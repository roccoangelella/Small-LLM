# Decision log

Each ADR records one durable choice, its context, alternatives, outcome, and consequences. Accepted ADRs are not rewritten to hide changed reasoning; create a new ADR that supersedes the old one.

## Status meanings

- `proposed`: under discussion, not authorized.
- `accepted`: current durable decision.
- `superseded`: replaced by a later ADR.
- `rejected`: considered and explicitly not chosen.

## Accepted ADRs

- [`0114-run-deep-decay-100m-10b-on-modal-h100.md`](0114-run-deep-decay-100m-10b-on-modal-h100.md)
- [`0113-use-superior-reasoning-stage2-for-rsft-scaling.md`](0113-use-superior-reasoning-stage2-for-rsft-scaling.md)
- [`0112-scale-rsft-with-nested-1-2-4-percent-superior-reasoning-corpora.md`](0112-scale-rsft-with-nested-1-2-4-percent-superior-reasoning-corpora.md)
- [`0111-allow-production-rsft-epoch-count.md`](0111-allow-production-rsft-epoch-count.md)
- [`0110-disable-xet-for-kaggle-rsft-checkpoint-upload.md`](0110-disable-xet-for-kaggle-rsft-checkpoint-upload.md)
- [`0109-bind-100m-rsft-to-profile-specific-hf-repository.md`](0109-bind-100m-rsft-to-profile-specific-hf-repository.md)
- [`0108-promote-expanded-rsft-corpus-to-kaggle-default.md`](0108-promote-expanded-rsft-corpus-to-kaggle-default.md)
- [`0106-resume-expanded-rsft-with-curation-v2-and-keeper-only-gemini.md`](0106-resume-expanded-rsft-with-curation-v2-and-keeper-only-gemini.md)
- [`0001-use-structured-markdown-project-memory.md`](0001-use-structured-markdown-project-memory.md)
- [`0002-freeze-eval-core-v1-and-unified-cli.md`](0002-freeze-eval-core-v1-and-unified-cli.md)
- [`0003-defer-architecture-baselines-until-larger-models.md`](0003-defer-architecture-baselines-until-larger-models.md)
- [`0004-run-100m-in-one-session-with-250-step-durability.md`](0004-run-100m-in-one-session-with-250-step-durability.md)
- [`0005-adapt-gdn2-chunks-to-decay-span.md`](0005-adapt-gdn2-chunks-to-decay-span.md)
- [`0006-calibrate-fp16-loss-scale-before-failing-block.md`](0006-calibrate-fp16-loss-scale-before-failing-block.md)
- [`0007-render-teacher-forced-examples-as-readable-ground-truth.md`](0007-render-teacher-forced-examples-as-readable-ground-truth.md)
- [`0009-start-500m-at-microbatch-4-with-250-step-durability.md`](0009-start-500m-at-microbatch-4-with-250-step-durability.md)
- [`0010-self-provision-eval-core-from-main-evaluator.md`](0010-self-provision-eval-core-from-main-evaluator.md)
- [`0011-publish-standalone-gated-delta-rule-package.md`](0011-publish-standalone-gated-delta-rule-package.md)
- [`0013-rename-public-gdr2-repository.md`](0013-rename-public-gdr2-repository.md)
- [`0014-simplify-public-gdr2-repository-docs.md`](0014-simplify-public-gdr2-repository-docs.md)
- [`0015-scrub-small-llm-references-from-public-gdr2.md`](0015-scrub-small-llm-references-from-public-gdr2.md)
- [`0016-qualify-fla-gdn2-before-changing-decay.md`](0016-qualify-fla-gdn2-before-changing-decay.md)
- [`0017-use-latest-pointer-for-10m-prompt-comparison.md`](0017-use-latest-pointer-for-10m-prompt-comparison.md)
- [`0018-integrate-fla-gdn2-as-checkpoint-compatible-cuda-backend.md`](0018-integrate-fla-gdn2-as-checkpoint-compatible-cuda-backend.md)
- [`0019-resume-500m-checkpoint-with-fla-gdn2-execution.md`](0019-resume-500m-checkpoint-with-fla-gdn2-execution.md)
- [`0020-qualify-fla-gdn2-with-full-fp32-kernel-execution.md`](0020-qualify-fla-gdn2-with-full-fp32-kernel-execution.md)
- [`0021-qualify-fla-gdn2-v052-and-resume-step4000.md`](0021-qualify-fla-gdn2-v052-and-resume-step4000.md)
- [`0023-run-2b-20m-probe-via-vps-kaggle-dataset.md`](0023-run-2b-20m-probe-via-vps-kaggle-dataset.md)
- [`0025-freeze-canonical-full-post-pretraining-prompt-suite.md`](0025-freeze-canonical-full-post-pretraining-prompt-suite.md)
- [`0027-use-500m-schema-gains-to-justify-fixed-20m-token-scaling-through-2b.md`](0027-use-500m-schema-gains-to-justify-fixed-20m-token-scaling-through-2b.md)
- [`0028-use-one-profile-driven-launcher-for-publication-and-training.md`](0028-use-one-profile-driven-launcher-for-publication-and-training.md)
- [`0030-consolidate-kaggle-profile-wrappers-behind-one-runtime.md`](0030-consolidate-kaggle-profile-wrappers-behind-one-runtime.md)
- [`0031-govern-project-memory-with-progressive-disclosure.md`](0031-govern-project-memory-with-progressive-disclosure.md)
- [`0032-scale-sft-budget-with-pretraining-and-qualify-on-500m-first.md`](0032-scale-sft-budget-with-pretraining-and-qualify-on-500m-first.md)
- [`0033-use-comprehensive-post-sft-qualification-and-pretraining-cadence.md`](0033-use-comprehensive-post-sft-qualification-and-pretraining-cadence.md)
- [`0034-make-sft-data-publication-machine-agnostic.md`](0034-make-sft-data-publication-machine-agnostic.md)
- [`0035-retire-completed-fla-investigation-scripts-from-kaggle.md`](0035-retire-completed-fla-investigation-scripts-from-kaggle.md)
- [`0037-consolidate-dataset-profile-tools-and-retire-one-off-qualification-code.md`](0037-consolidate-dataset-profile-tools-and-retire-one-off-qualification-code.md)
- [`0038-stream-local-chat-tokens-as-generated.md`](0038-stream-local-chat-tokens-as-generated.md)
- [`0039-use-modal-for-future-gpu-training.md`](0039-use-modal-for-future-gpu-training.md)
- [`0041-use-block64-modal-corpus-and-probe-microbatch-16-32-48-64.md`](0041-use-block64-modal-corpus-and-probe-microbatch-16-32-48-64.md)
- [`0043-prepare-modal-block64-corpus-on-vps.md`](0043-prepare-modal-block64-corpus-on-vps.md)
- [`0044-publish-100m-2b-final-model-to-hugging-face.md`](0044-publish-100m-2b-final-model-to-hugging-face.md)
- [`0048-verify-modal-dataset-in-active-workspace.md`](0048-verify-modal-dataset-in-active-workspace.md)
- [`0049-make-modal-launch-logs-concise-and-unambiguous.md`](0049-make-modal-launch-logs-concise-and-unambiguous.md)
- [`0051-qualify-exact-batch-dual-t4-ddp-before-kaggle-adoption.md`](0051-qualify-exact-batch-dual-t4-ddp-before-kaggle-adoption.md)
- [`0053-stream-10b-through-one-gib-hf-shards-and-cpu-stage-before-h100.md`](0053-stream-10b-through-one-gib-hf-shards-and-cpu-stage-before-h100.md)
- [`0054-retire-google-drive-for-new-dataset-durability.md`](0054-retire-google-drive-for-new-dataset-durability.md)
- [`0055-unify-modal-checkpoints-on-hf-model-repository.md`](0055-unify-modal-checkpoints-on-hf-model-repository.md)
- [`0056-adopt-exact-batch-dual-t4-ddp-for-kaggle-only.md`](0056-adopt-exact-batch-dual-t4-ddp-for-kaggle-only.md)
- [`0057-use-standard-wsd-for-100m-10b.md`](0057-use-standard-wsd-for-100m-10b.md)
- [`0058-produce-10b-shards-concurrently-with-modal-training.md`](0058-produce-10b-shards-concurrently-with-modal-training.md)
- [`0059-run-supplementary-sampled-three-way-full-evaluation.md`](0059-run-supplementary-sampled-three-way-full-evaluation.md)
- [`0060-require-live-modal-hf-smoke-before-100m-10b.md`](0060-require-live-modal-hf-smoke-before-100m-10b.md)
- [`0061-add-beam-as-an-alternate-single-gpu-training-provider.md`](0061-add-beam-as-an-alternate-single-gpu-training-provider.md)
- [`0062-default-beam-training-to-serverless-rtx5090.md`](0062-default-beam-training-to-serverless-rtx5090.md)
- [`0063-document-local-chat-cli-usage-in-source.md`](0063-document-local-chat-cli-usage-in-source.md)
- [`0064-allow-stable-pretrained-artifacts-in-local-chat.md`](0064-allow-stable-pretrained-artifacts-in-local-chat.md)
- [`0065-cap-beam-startup-probe-at-microbatch-16.md`](0065-cap-beam-startup-probe-at-microbatch-16.md)
- [`0067-keep-100m-2b-sft-at-4-percent-on-dual-t4.md`](0067-keep-100m-2b-sft-at-4-percent-on-dual-t4.md)
- [`0069-own-beam-launch-namespace-in-synced-checkout.md`](0069-own-beam-launch-namespace-in-synced-checkout.md)
- [`0070-use-vps-fed-beam-volume-for-10b-dataset-production.md`](0070-use-vps-fed-beam-volume-for-10b-dataset-production.md)
- [`0071-run-full-100m-10b-with-concurrent-5b-evaluation.md`](0071-run-full-100m-10b-with-concurrent-5b-evaluation.md)
- [`0072-pin-beam-client-0207-for-live-gateway.md`](0072-pin-beam-client-0207-for-live-gateway.md)
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

## Superseded ADRs

- [`0099-run-deep-decay-100m-10b-on-kaggle-dual-t4.md`](0099-run-deep-decay-100m-10b-on-kaggle-dual-t4.md) — superseded by ADR 0114, which retains the ADR-0095 schedule and namespace but moves execution to one exact Modal H100 with microbatch-16 slicing inside the same global block 64.
- [`0050-scale-100m-to-fresh-10b-with-5b-capability-gate.md`](0050-scale-100m-to-fresh-10b-with-5b-capability-gate.md) — superseded by ADR 0071 after the user authorized the full 10B run with a concurrent, non-blocking approximately-5B Kaggle evaluation.
- [`0068-make-plain-uv-sync-install-complete-runtime.md`](0068-make-plain-uv-sync-install-complete-runtime.md) — its complete-runtime rule is retained, but ADR 0072 replaces the gateway-rejected Beam 0.2.201 pin with 0.2.207.
- [`0066-run-100m-2b-sft-at-10-percent-on-dual-t4.md`](0066-run-100m-2b-sft-at-10-percent-on-dual-t4.md) — superseded by ADR 0067, which retains dual-T4 execution but restores the 4%-of-parent SFT budget.

- [`0008-run-500m-final-20m-data-scaling-probe.md`](0008-run-500m-final-20m-data-scaling-probe.md) — its 500M run decision was executed, but its claim that 500M would be the final 20M data-scaling probe was superseded by later scaling decisions.
- [`0012-bind-public-gdr2-repository-identity.md`](0012-bind-public-gdr2-repository-identity.md) — superseded by ADR 0013 after the repository was renamed.
- [`0022-run-1b-20m-probe-via-vps-kaggle-dataset.md`](0022-run-1b-20m-probe-via-vps-kaggle-dataset.md) — superseded before execution by ADR 0023, which changes the target to 2B tokens while retaining VPS build plus private Kaggle attachment.
- [`0024-freeze-canonical-questions-only-prompt-test-settings.md`](0024-freeze-canonical-questions-only-prompt-test-settings.md) — superseded by ADR 0025 after the user clarified that the reusable canonical comparison should run the full qualitative prompt suite.
- [`0026-prune-superseded-one-off-kaggle-diagnostics.md`](0026-prune-superseded-one-off-kaggle-diagnostics.md) — superseded by ADR 0035 after the completed FLA investigation executables were re-audited and retired while active runtime/test dependencies were retained.
- [`0029-limit-pre-2b-kaggle-cleanup-to-dead-wrappers-and-dispatch-fixes.md`](0029-limit-pre-2b-kaggle-cleanup-to-dead-wrappers-and-dispatch-fixes.md) — superseded by ADR 0030 after the user explicitly authorized consolidating profile wrappers/overlays before the 2B run.
- [`0036-add-local-completed-sft-chat-cli.md`](0036-add-local-completed-sft-chat-cli.md) — superseded by ADR 0064, which retains strict completed-SFT validation but expands the CLI to explicitly registered stable pretrained artifacts, starting with 100M / 2B.
- [`0040-launch-100m-2b-pretraining-on-modal.md`](0040-launch-100m-2b-pretraining-on-modal.md) — its Modal launch authorization is retained by ADR 0041, which replaces the 16-sequence dataset/optimizer block and 4/8/16 probe with byte-preserving block 64 and 16/32/48/64 qualification.
- [`0042-derive-modal-block64-corpus-on-kaggle.md`](0042-derive-modal-block64-corpus-on-kaggle.md) — superseded by ADR 0043, which keeps Kaggle only as the remote dataset source and moves download, verification, reblock, Modal upload, and launch control to the VPS.
- [`0045-run-periodic-hf-backups-only-while-modal-training-is-live.md`](0045-run-periodic-hf-backups-only-while-modal-training-is-live.md) — superseded by ADR 0046, which integrated Hugging Face publication and restore directly into the Modal training path.
- [`0046-use-rolling-hf-as-modal-cross-workspace-checkpoint-transport.md`](0046-use-rolling-hf-as-modal-cross-workspace-checkpoint-transport.md) — superseded before production use by ADR 0047, which keeps the integrated HF resume design but moves mutable checkpoints from a Git-backed model repository to a Hugging Face Storage Bucket.
- [`0047-use-hf-storage-bucket-for-modal-cross-workspace-checkpoints.md`](0047-use-hf-storage-bucket-for-modal-cross-workspace-checkpoints.md) — superseded by ADR 0055, which returns Modal checkpoint durability to the unified Hugging Face model repository while keeping Storage Buckets for dataset object transport.
- [`0052-evaluate-modal-rolling-checkpoints-directly-from-hf-bucket.md`](0052-evaluate-modal-rolling-checkpoints-directly-from-hf-bucket.md) — superseded by ADR 0055 because stable `models/...` artifacts and model-repository `run/...` checkpoints no longer require a separate bucket-only evaluation path.

Use [`template.md`](template.md) for new decisions. Historical omnibus decision registers are retained under [`../archive/decision_registers/`](../archive/decision_registers/decisions_and_ablations.md) but are no longer the preferred format for new choices.
