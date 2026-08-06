# Small LLM Technical Documentation

The files in this directory are the sole source of truth for the Small LLM project's goals, decisions, technical contracts, current status, and open questions.

## Documents

- [`project_overview.md`](project_overview.md): project goal, scope, resource assumptions, development strategy, and documentation policy.
- [`project_status.md`](project_status.md): current phase, completed foundations, operational gates, immediate next steps, and open decisions.
- [`model_architecture.md`](model_architecture.md): decoder block structure, mixers, normalization, position handling, FFNs, embeddings, and output path.
- [`model_geometry.md`](model_geometry.md): scalable model family, frozen smoke and approximately 100M configurations, parameter accounting, and hardware-friendly dimensions.
- [`dataset_and_tokenization.md`](dataset_and_tokenization.md): tokenizer, pinned source, content policy, exact mixture, scheduler, packing, cache, durability, Drive integration, and model-facing data contract.
- [`training_and_evaluation.md`](training_and_evaluation.md): trainer constraints, experiment ladder, joint checkpoint contract, instrumentation, and unresolved training/evaluation choices.
- [`decisions_and_ablations.md`](decisions_and_ablations.md): frozen defaults, planned controlled ablations, replaced decisions, and decision standards.
- [`eval_core_v1_design.md`](eval_core_v1_design.md): frozen stratified holdout size, leakage contract, intrinsic scorecard, prompt-suite integration, and architecture-comparison timing.
- [`eval_core_v1_runbook.md`](eval_core_v1_runbook.md): installed build, verification, fast-suite, and full-suite commands plus result and operational contracts.
- [`post_pretraining_prompt_suite.md`](post_pretraining_prompt_suite.md): repository-native qualitative generation procedure for verified remote base checkpoints.
- [`20m_post_pretraining_checkpoint_selection.md`](20m_post_pretraining_checkpoint_selection.md): legacy final-checkpoint selection rule for the completed 20M run.
- [`20m_post_pretraining_qualitative_results.md`](20m_post_pretraining_qualitative_results.md): first prompt-suite outputs, measured summary, interpretation, and accepted smoke-scale learning-signal verdict.

## Maintenance rule

When a project decision or operational fact changes:

1. update the relevant topic document;
2. update [`project_status.md`](project_status.md) when current state, next steps, or open questions change;
3. update [`decisions_and_ablations.md`](decisions_and_ablations.md) when a default is frozen, replaced, or promoted to an ablation;
4. state what changed, why it changed, and which benchmark, operational result, or new requirement justified it.

Do not silently overwrite historical reasoning. The documentation should evolve continuously with implementation and experiments.
