# eval_core_v1 and Pretraining Qualification v2 Runbook

## Purpose

Describe the staged evaluation v2 target for running the frozen intrinsic
evaluation together with the external capability and expanded base-prompt
layers.

ADR 0140 has landed additive implementation modules and documentation. The final
in-place replacement of active evaluator entrypoints is pending a tested
follow-up patch.

## Environment

Install the normal model environment plus the isolated evaluator dependency:

```bash
python -m pip install -r requirements-eval.txt
```

`lm-evaluation-harness` is pinned separately so evaluation tooling does not
change the training lock.

## Full qualification target

```bash
python -m trainer.eval_entrypoint full \
  --repo-id <repo> \
  --run-id <run> \
  --pointer best \
  --output-json artifacts/<run>-pretraining-qualification-v2.json
```

When the v2 entrypoint wiring lands, the full report should contain:

- frozen `eval_core_v1`;
- full six-task L20 conditional-likelihood evaluation;
- 100 mechanically scored base prompts;
- 20 qualitative continuations;
- greedy and canonical sampled prompt views.

Prompt budgets are native per case.

## Fast diagnostic target

```bash
python -m trainer.eval_entrypoint fast \
  --repo-id <repo> \
  --run-id <run> \
  --output-json artifacts/<run>-pretraining-fast-v2.json
```

Fast mode should limit external tasks to 100 examples each and use a reduced
prompt subset. It is useful for smoke testing but is not a final qualification.

## Interpretation

Read `read_me_first` and `headline_summary` first.

- loss / perplexity / BPB: lower is better;
- top-k accuracy: higher is better;
- L20 mean-6: higher is better;
- base-prompt accuracy: higher is better.

EOS termination is intentionally absent from pretraining scoring.
Teacher-forced confidence remains a separate diagnostic.
