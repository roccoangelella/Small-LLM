# 0134 — Build all-model answer corpus

Date: 2026-08-31
Status: Accepted

## Decision

Produce document-ready answer-corpus artifacts for every Small-LLM model endpoint trained so far, using both deterministic greedy outputs and sampled outputs.

The corpus should cover:

- completed pretrained endpoints;
- active pretrained checkpoints when they are used as comparison points;
- SFT checkpoints, including the canonical 100M/2B 10% peak-through-3000 SFT checkpoint;
- R-SFT checkpoints when the question is post-SFT behavior or reasoning progression.

## Operational interpretation

For pretrained base models, run the frozen `eval_core_v1` full evaluator separately under greedy-native decoding and ADR-0136 sampled-native decoding when prompt outputs must be harvested from the full-eval JSON. Do not use the separate ADR-0025 greedy-32 cap unless the exact greedy-32 qualitative protocol is explicitly required.

For ordinary SFT qualification, use the comprehensive SFT evaluator when full SFT metrics are required; its sampled qualitative branch now follows ADR 0136 (`T=1.0`, `top_p=1.0`, `top_k=0`, native budgets). For all-model answer-corpus work that explicitly excludes greedy-32, run the full base evaluator in greedy-native and sampled-native modes against the SFT checkpoint.

For R-SFT qualification, use the comprehensive R-SFT evaluator because its JSON includes S0-to-R-SFT comparison axes, deterministic prompt regressions, standard sampled-native regressions, and reasoning samples.

## Notes

This decision concerns evaluation/reporting artifacts only. It does not change model selection, training hyperparameters, checkpoint publication policy, or canonical eval-core metrics.