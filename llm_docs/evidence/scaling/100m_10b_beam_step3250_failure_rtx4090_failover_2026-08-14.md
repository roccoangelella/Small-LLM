---
status: observed
date: 2026-08-14
---

# 100M / 10B step-3250 failure and RTX4090 failover

The replacement RTX5090 segment remained numerically healthy through update
3,250. Its boundary metrics were finite:

```text
step: 3250
target tokens/s: 41,754.4268
overflow events: 0
validation blocks: 16
validation loss: 3.7223885756
validation perplexity: 41.3630750
```

The container disappeared immediately after validation while the next local
checkpoint was beginning. Beam finalized GPU task
`7e92c8a8-d893-4d49-be00-500db8b5e7c9` as `ERROR` at 11:48:49 UTC and W&B
marked the run crashed. Beam Volume contained an empty
`.step-00003250.crk3st56` staging directory but no finalized step-3,250
checkpoint. No trainer traceback, CUDA error, overflow, or nonfinite metric was
observed. The Beam CLI task-log endpoint itself failed its TLS handshake, so the
provider-side exception was unavailable.

Both Beam Volume and the Hugging Face model repository independently retained
the complete `step-00003000`. HF `run/100m-10b-data-001/latest.json` pointed to
that checkpoint and recorded the 913,877,803-byte `trainer_state.pkl` hash.

An unchanged RTX5090 relaunch passed import, dataset staging, and visibility
gates, then GPU task `e860deda-c733-4bef-ba75-b366dc32cf6a` failed after 167
seconds before it emitted the trainer command or contacted W&B. This failure on
a clean exact restore, before model execution began, strengthened the evidence
for Beam RTX5090 worker/Volume infrastructure instability rather than a
numerical training failure.

The next retry used Beam's already-supported RTX4090 lane while retaining exact
source commit `1f9dff920ecc45ce2fdb43fd875514a18391273d`, checkpoint
`step-00003000`, microbatch 4, FP16, dataset order, optimizer and scaler state,
WSD position, W&B identity, and the uncapped terminal horizon. GPU task
`15e6f9dc-5645-4799-8867-da3d3805119c` ran on a different worker and machine.
The trainer launched with 73,294 remaining updates and W&B resume policy
`must`; W&B returned to `running` on the unchanged run.

Production advanced through at least update 3,009. Step 3,009 reported finite
loss 3.639936, gradient norm 1.043473, scaler 131,072, and no observed overflow;
steady RTX4090 throughput was approximately 31-33 thousand target tokens/s.
