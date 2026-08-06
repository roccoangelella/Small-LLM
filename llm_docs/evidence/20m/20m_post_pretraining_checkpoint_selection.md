# 20M Post-Pretraining Checkpoint Selection

_Last updated: 2026-08-05_

## Decision

The completed `20m-qualification-dataset-001` run predates validation-selected remote `best.json` publication. Its post-pretraining qualitative prompt suite must therefore evaluate the final remotely verified training snapshot rather than a nonexistent best pointer.

The selected checkpoint is:

```text
run/20m-qualification-dataset-001/checkpoints/step-00000306/last
```

The prompt suite must select it through the run's authoritative `latest.json` pointer by passing:

```text
--run-id 20m-qualification-dataset-001 --pointer latest
```

This is an explicit exception for the completed 20M qualification run. Future substantive runs that publish held-out validation metrics should continue to use `--pointer best` so qualitative evaluation targets the lowest-validation-loss checkpoint.

## Legacy model configuration

This checkpoint was created before native checkpoints embedded `model_config`. The prompt suite must therefore be supplied the frozen 20M model configuration explicitly. The expected geometry is the smoke configuration with:

- architecture: `gdn2_hybrid`;
- context length: 2,048;
- GDN chunk size: 32;
- the otherwise frozen `ModelConfig.smoke()` values.

Generate the JSON from the repository's own `ModelConfig` implementation rather than maintaining a hand-copied configuration.
