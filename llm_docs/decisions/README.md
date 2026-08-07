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
- [`0008-run-500m-final-20m-data-scaling-probe.md`](0008-run-500m-final-20m-data-scaling-probe.md)
- [`0009-start-500m-at-microbatch-4-with-250-step-durability.md`](0009-start-500m-at-microbatch-4-with-250-step-durability.md)
- [`0010-self-provision-eval-core-from-main-evaluator.md`](0010-self-provision-eval-core-from-main-evaluator.md)
- [`0011-publish-standalone-gated-delta-rule-package.md`](0011-publish-standalone-gated-delta-rule-package.md)
- [`0013-rename-public-gdr2-repository.md`](0013-rename-public-gdr2-repository.md)
- [`0014-simplify-public-gdr2-repository-docs.md`](0014-simplify-public-gdr2-repository-docs.md)
- [`0015-scrub-small-llm-references-from-public-gdr2.md`](0015-scrub-small-llm-references-from-public-gdr2.md)

## Superseded ADRs

- [`0012-bind-public-gdr2-repository-identity.md`](0012-bind-public-gdr2-repository-identity.md) — superseded by ADR 0013 after the repository was renamed.

Use [`template.md`](template.md) for new decisions. Historical omnibus decision registers are retained under [`../archive/decision_registers/`](../archive/decision_registers/decisions_and_ablations.md) but are no longer the preferred format for new choices.
