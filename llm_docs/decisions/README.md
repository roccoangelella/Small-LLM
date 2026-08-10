# Decision log

Each ADR records one durable choice, its context, alternatives, outcome, and consequences. Accepted ADRs are not rewritten to hide changed reasoning; create a new ADR that supersedes the old one.

## Status meanings

- `proposed`: under discussion, not authorized.
- `accepted`: current durable decision.
- `superseded`: replaced by a later ADR.
- `rejected`: considered and explicitly not chosen.

## Accepted ADRs

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
- [`0026-prune-superseded-one-off-kaggle-diagnostics.md`](0026-prune-superseded-one-off-kaggle-diagnostics.md)
- [`0027-use-500m-schema-gains-to-justify-fixed-20m-token-scaling-through-2b.md`](0027-use-500m-schema-gains-to-justify-fixed-20m-token-scaling-through-2b.md)
- [`0028-use-one-profile-driven-launcher-for-publication-and-training.md`](0028-use-one-profile-driven-launcher-for-publication-and-training.md)
- [`0029-limit-pre-2b-kaggle-cleanup-to-dead-wrappers-and-dispatch-fixes.md`](0029-limit-pre-2b-kaggle-cleanup-to-dead-wrappers-and-dispatch-fixes.md)

## Superseded ADRs

- [`0008-run-500m-final-20m-data-scaling-probe.md`](0008-run-500m-final-20m-data-scaling-probe.md) — its 500M run decision was executed, but its claim that 500M would be the final 20M data-scaling probe was superseded by later scaling decisions.
- [`0012-bind-public-gdr2-repository-identity.md`](0012-bind-public-gdr2-repository.md) — superseded by ADR 0013 after the repository was renamed.
- [`0022-run-1b-20m-probe-via-vps-kaggle-dataset.md`](0022-run-1b-20m-probe-via-vps-kaggle-dataset.md) — superseded before execution by ADR 0023, which changes the target to 2B tokens while retaining VPS build plus private Kaggle attachment.
- [`0024-freeze-canonical-questions-only-prompt-test-settings.md`](0024-freeze-canonical-questions-only-prompt-test-settings.md) — superseded by ADR 0025 after the user clarified that the reusable canonical comparison should run the full qualitative prompt suite.

Use [`template.md`](template.md) for new decisions. Historical omnibus decision registers are retained under [`../archive/decision_registers/`](../archive/decision_registers/decisions_and_ablations.md) but are no longer the preferred format for new choices.
