# Post-Pretraining Prompt Evaluation

The old fixed-length qualitative protocol is retired. Canonical prompt
qualification now lives inside pretraining evaluation v2.

Use:

```bash
python -m trainer.eval_entrypoint full \
  --repo-id <repo> \
  --run-id <run> \
  --output-json artifacts/<run>-pretraining-qualification-v2.json
```

The current prompt layer contains 100 mechanically scored base-model cases and
20 readable qualitative continuations. Every case uses its own native
generation budget.

Canonical decoding:

```text
greedy:  temperature=0, top_p=1, top_k=0, seed=17
sampled: temperature=1, top_p=1, top_k=0, seed=17
```

The standalone `trainer.post_pretraining_prompt_suite` module remains a
low-level/debug utility for the original prompt inventory. It is not the
canonical model qualification protocol.
