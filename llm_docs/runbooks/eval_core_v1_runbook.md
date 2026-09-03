# eval_core_v1 and Pretraining Qualification v2 Runbook

## Purpose

Run the frozen intrinsic evaluation together with the current external
capability and expanded base-prompt layers.

## Environment

Install the normal model environment plus the isolated evaluator dependency:

```bash
python -m pip install -r requirements-eval.txt
```

`lm-evaluation-harness` is pinned separately so evaluation tooling does not
change the training lock.

## Full qualification

```bash
python -m trainer.eval_entrypoint full \
  --repo-id <repo> \
  --run-id <run> \
  --pointer best \
  --output-json artifacts/<run>-pretraining-qualification-v2.json
```

If `eval_core_v1` is not attached, the entrypoint self-provisions and verifies
the frozen corpus.

The full report contains:

- frozen `eval_core_v1`;
- full six-task L20 conditional-likelihood evaluation;
- 100 mechanically scored base prompts;
- 20 qualitative continuations;
- greedy and canonical sampled prompt views.

Prompt budgets are native per case.

## Fast diagnostic

```bash
python -m trainer.eval_entrypoint fast \
  --repo-id <repo> \
  --run-id <run> \
  --output-json artifacts/<run>-pretraining-fast-v2.json
```

Fast mode limits external tasks to 100 examples each and uses a reduced prompt
subset. It is useful for smoke testing but is not a final qualification.

## Interpretation

Read `read_me_first` and `headline_summary` first.

- loss / perplexity / BPB: lower is better;
- top-k accuracy: higher is better;
- L20 mean-6: higher is better;
- base-prompt accuracy: higher is better.

EOS termination is intentionally absent from pretraining scoring.
Teacher-forced confidence remains a separate diagnostic.
