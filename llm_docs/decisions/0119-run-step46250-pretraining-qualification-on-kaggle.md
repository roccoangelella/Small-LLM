---
status: accepted
date: 2026-08-24
---

# Run the live 100M/10B pretraining qualification on Kaggle

The live `100m-10b-deep-decay-from-step15500` trajectory is still training. Run its intermediate pretraining qualification on Kaggle rather than on the VPS, resolving the newest fully published Hugging Face snapshot through `--pointer latest` at evaluation start. A repository revision does not need to be pinned unless an exact historical checkpoint must be reproduced.

Use the standard pretrained-model evaluation matrix against that live snapshot:

1. frozen `eval_core_v1` full intrinsic evaluation;
2. canonical greedy-32 qualitative suite (`temperature=0`, `top_p=1`, `top_k=0`, seed 17, one sample, global `max_new_tokens=32`);
3. supplementary wider sampled suite (`temperature=1.0`, `top_p=0.9`, `top_k=20`, seed 17, one sample, native prompt budgets);
4. teacher-forced held-out confidence diagnostic.

The first Kaggle attempt exposed a compatibility bug in `trainer/post_pretraining_prompt_suite.py`: the evaluator used raw `pickle.load()` on `trainer_state.pkl`, while current live checkpoints are serialized with streamed `torch.save`. The evaluator must use the existing `trainer.state.load_trainer_state_file` compatibility loader so both historical plain-pickle and current streamed checkpoints remain evaluable. This repair applies to intrinsic evaluation and both qualitative/teacher-forced paths because they share the same model loader.
