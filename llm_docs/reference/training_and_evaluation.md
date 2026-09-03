# Training and Evaluation Reference

## Evaluation v2 staging state

ADR 0140 adopts evaluation v2 as the next canonical target and lands additive
implementation modules. The final in-place replacement of active evaluator
entrypoints is pending a tested follow-up patch.

## Pretraining qualification target

The evaluation v2 target for `small-llm-eval` / `python -m trainer.eval_entrypoint`
contains:

- `eval_core_v1`: frozen intrinsic next-token metrics;
- L20 six-task zero-shot conditional-likelihood evaluation;
- expanded base prompt benchmark: 100 scored prompts + 20 qualitative continuations.

The L20 layer is pinned to `lm-evaluation-harness==0.4.12` and uses
ARC-Challenge, ARC-Easy, HellaSwag, LAMBADA OpenAI, PIQA and WinoGrande.
Generation is forbidden for that layer.

Install evaluator-only dependencies separately:

```bash
python -m pip install -r requirements-eval.txt
```

Qualitative decoding target:

```text
greedy:  temperature=0, top_p=1, top_k=0, seed=17
sampled: temperature=1, top_p=1, top_k=0, seed=17
budget:  native per prompt
```

EOS termination is not a pretraining metric. Teacher-forced confidence remains
a separate diagnostic (`trainer.teacher_forced_diagnostic`).

## SFT qualification target

The evaluation v2 target for `small-llm-sft-eval` / `python -m post_training.sft.eval_suite`
compares immutable parent and SFT checkpoints on:

- `eval_core_v1` retention;
- masked SFT validation/test loss as diagnostics only;
- legacy SFT behavior v1 for historical continuity;
- SFT Behavior v2 as primary instruction-following evidence;
- native-budget base qualitative regressions.

Behavior v2 contains 180 semantic tasks and four paired levels (L0-L3),
producing 720 cases. The held-out qualification partition has 240 cases.
L1/L2/L3 conditional compliance uses only tasks whose L0 semantic answer was
correct.

Greedy is primary. Sampled robustness uses
`temperature=1`, `top_p=1`, `top_k=0` with seeds 17, 18 and 19.

## Report reading order

Canonical v2 JSON should preserve semantic key order:

1. `read_me_first`;
2. `headline_summary`;
3. `protocol`;
4. checkpoint/bundle identity;
5. detailed metrics and per-case evidence;
6. report hash.

Do not infer instruction-following success from teacher-forced SFT loss alone.
Do not infer pretraining quality from EOS behavior.
