---
status: accepted
date: 2026-09-08
supersedes: null
---

# 0157 — Paired MoE pilot with a committed selection controller

## Context and problem statement

The `moe-8e-top1` branch has an unbalanced control but needs an executable, observable M0/M1 comparison. Edo authorized local plan/code corrections and local commits on 2026-09-08; paid runs and remote publication remain separate. This decision extends branch ADRs 0154–0156: their no-balancing rule defines M0; M1 is the explicit additional arm. The dense endpoint is historical context; D100 only measures systems behavior.

## Considered options

- Keep only the unbalanced M0 control.
- Add a score-domain sign controller while preserving the current softmax gate.
- Reproduce the paper's proportional-error softmax variant or change the gate to sigmoid.

## Decision outcome

Choose **one explicit sign-controller intervention**, serialized as MoE config version 2:

- M0: `load_balancing=none`, `balancing_step_size=0`.
- M1: `loss_free_sign`, gamma `1e-3` as a transferred starting hypothesis.
- In FP32, `p=softmax(z)`, select `argmax(p+b)`, combine using the original selected `p`. The nontrainable bias never enters the gate or z-loss.
- Initialize eight biases per layer to zero. After each accepted full optimizer update, set `b += gamma * sign(mean(counts)-counts)`. All microbatches and retries of an update share the same bias. Eval never advances it.
- Commit bias, cumulative routed-position exposure and nonempty expert-update counters with the model state. There is no pending controller operation at a checkpoint: the next bias is already applied, so storing previous raw counts is unnecessary. Model/trainer config, optimizer, schedule, cursor and RNG retain the joint checkpoint identity checks.
- This adapts Eq. 3 / Algorithm 1 to the existing softmax router. It is **not** the paper's Appendix C.2 proportional-error softmax experiment, and gamma is not a success threshold.
- Start M0/M1 from scratch with matched seed, stream, precision, microbatch, full manifest-derived WSD schedule and update cap. No quality cutoff, HPO, upcycling or extra dense endpoint is introduced.

Use a real `python -m MOE_model` child process. Provider pilots retain selected joint checkpoints in arm-specific volume namespaces and bind their source/data/recipe identity before execution. Retry caps are absolute successful-update caps. No automatic change of precision or microbatch is permitted.

Observation is optional in the shared trainer: local identity/segment manifests, JSONL updates, selected checkpoint probes and bounded profiles. Fixed probes contain token/label/mask identity, FP32 per-target CE, router logits/assignments/original gate and checkpoint binding. Full LM logits are opt-in. Sample expert weight norms and **post-clipping** gradient norms before validation clears gradients; derive weight changes offline between checkpoints, explicitly as interval changes. Routed-position counts and loss-bearing target counts have different denominators when masks are present.

## Consequences

### Positive

The changed scientific variable is visible, checkpointed and testable. Failed norm/scale attempts cannot step the optimizer or advance controller/data counters; rejected candidate LR is restored. Fresh-process resume and observational artifacts use the same production loop.

### Negative or limiting

The initial MoE gate near 1/8 changes FFN amplitude relative to dense, so dense/MoE does not isolate capacity. A single seed gives indicative evidence. CPU tests do not qualify CUDA/FLA precision, throughput or memory. Profiles and checkpoint probes perturb timing; exclude them from clean update windows and report their overhead and full wall time separately. Provider invoices, CPU/RAM, lost attempts and storage remain necessary for economic comparisons. Config-v1 MoE checkpoints are intentionally not silently reinterpreted as v2.

## Validation

Run controller, overflow, data-path, evaluation-identity and real CLI tests. Compare uninterrupted training with checkpoint/resume at the same boundary, including optimizer/RNG/controller state. Verify gamma-zero output/gradient equivalence and probe non-mutation. Before a pilot, run the CUDA precision check and a bounded provider checkpoint/resume qualification on the exact image/data. See the runbook for commands and the remaining launch inputs.

## Links

- [Pilot runbook](../runbooks/moe-paired-pilot.md)
- [Loss-free balancing paper](https://arxiv.org/html/2408.15664v1)
- [PyTorch AMP](https://docs.pytorch.org/docs/2.10/amp.html)
