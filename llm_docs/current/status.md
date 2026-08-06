---
status: current
last_reviewed: 2026-08-06
---

# Current project status

## Active experiment

The approximately-20M-parameter GDN-2 hybrid is being trained on the fixed approximately-100M-token dataset.

```text
model parameters: 20,637,592
architecture: gdn2_hybrid
context: 2,048
precision: FP16
optimizer: hybrid Muon + AdamW
training venue: Kaggle NVIDIA T4
experiment: one pass over the fixed approximately-100M-token dataset
```

This file does not guess live step or token progress. Read the active W&B run and verified remote checkpoint pointer for exact progress.

## Accepted anchor

The completed 10M-token run remains the historical anchor:

```text
accepted source tokens: 10,000,662
optimizer updates: 306
final validation loss: 6.136690
final validation perplexity: 462.520157
FP16 overflow events: 0
```

## Frozen decisions affecting current work

- Continue the main GDN-2 hybrid through the 20M-model data-scaling stage.
- Do not run the matched all-attention or other mixer baseline yet.
- Revisit architecture comparisons when larger model versions are reached.
- Use the permanent stratified `eval_core_v1` fast/full suites and retain the existing prompt answers in the unified evaluator.

## Evaluation state

The repository contains:

```text
small-llm-eval-data build|verify
small-llm-eval fast|full
```

The code, manifest contract, streaming metrics, prompt integration, and offline tests are implemented. The production `eval_core_v1` corpus still needs to be built and its fast/full runtime measured on the T4 before it becomes an accepted evaluation artifact.

## Current source of truth

- Experiment procedure: [`../runbooks/20m_100m_runbook.md`](../runbooks/20m_100m_runbook.md)
- Evaluation procedure: [`../runbooks/eval_core_v1_runbook.md`](../runbooks/eval_core_v1_runbook.md)
- Model contract: [`../reference/model_architecture.md`](../reference/model_architecture.md)
- Dataset contract: [`../reference/dataset_and_tokenization.md`](../reference/dataset_and_tokenization.md)
- Durable decisions: [`../decisions/README.md`](../decisions/README.md)
