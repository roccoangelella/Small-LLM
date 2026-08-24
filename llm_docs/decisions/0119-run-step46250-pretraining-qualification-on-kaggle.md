---
status: accepted
date: 2026-08-24
---

# Run the step-46,250 pretraining qualification on Kaggle

The live `100m-10b-deep-decay-from-step15500` trajectory is still training. Evaluate checkpoint `step-00046250` as an intermediate pretraining qualification point on Kaggle rather than on the VPS.

Use the frozen `eval_core_v1` full suite plus the canonical post-pretraining greedy-32 prompt suite. Because the live `latest` pointer can advance while training continues, pin the Hugging Face repository revision whose `run/100m-10b-deep-decay-from-step15500/latest.json` resolves to `step-00046250` before launching evaluation.
