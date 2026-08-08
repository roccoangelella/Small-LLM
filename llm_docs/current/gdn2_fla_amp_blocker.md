---
status: current
last_reviewed: 2026-08-08
---

# Current blocker — FLA GDN-2 strong-decay AMP backward

The 20M/500M FLA migration is **not currently qualified for resumed training**.

Latest verified training checkpoint remains `step-00004000`. A real resume restored that checkpoint successfully but failed before update 4001 completed, so no checkpoint rollback is required.

After fixing the initial FP32/FP16 Triton input mismatch, the AMP-realistic full-layer probe (FP32 parameters + CUDA FP16 autocast) produced:

```text
normal_decay_amp: all layer outputs and parameter/input gradients PASS
strong_decay_-6_amp: layer output PASS, backward gradient comparison FAIL with NaNs
layer_forward_backward_parity: False
trainer_amp_contract_tested: True
```

This is the first test that actually inspects strong-decay gradients. The earlier standalone strong-decay forward+backward benchmark timed `.backward()` but did not validate stress-case gradient finiteness/parity.

The next isolated test is `kaggle/run_gdn2_fla_strong_decay_amp_probe.py`. It tests FLA with `disable_recompute=True`, matching the retained-intermediate mode used by the original standalone probe, and reports non-finite gradients separately for the adaptive reference and FLA candidate.

Do not resume the 500M run with FLA until that strong-decay AMP gradient gate passes.

Evidence: `llm_docs/evidence/gdn2_fla_amp_strong_decay_failure_2026-08-08.md`.
